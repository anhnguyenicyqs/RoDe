import os
import gc
import re
import json
import time
import math
import sys
import warnings
from pathlib import Path
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
import numpy as np
import matplotlib
# Use Agg backend to allow running headless (e.g. without UI window)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from PIL import Image

# Import local modules
from config import (
    LLM_MODEL_ID, SYSTEM_PROMPT, FEW_SHOT, OUTPUT_DIR,
    WALL_GAP, ITEM_GAP
)
from parser import RoomSpecParser
from layout import (
    auto_layout, ensure_must_have, _fallback_layout,
    get_item_center, get_rotated_corners, normalize_furniture_dimensions
)
from dxf_exporter import export_dxf
from renderer import create_depth_map, build_image_prompt, RoomRenderer

warnings.filterwarnings('ignore')

def query_llm_layout(spec, max_new_tokens=1600, max_retries=2):
    """
    Load Qwen 2.5-7B, query room layout as JSON, post-process, and unload from GPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"\n[LLM] Loading tokenizer and model: {LLM_MODEL_ID} (4-bit)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    llm.eval()
    
    W, L = spec['width'], spec['length']
    user_msg = json.dumps({
        'length': L, 'width': W, 'height': spec['height'],
        'function': spec['function'], 'style': spec['style'],
        'windows': spec['windows'], 'doors': spec['doors'],
        'budget': spec.get('budget', 'mid'),
    }, ensure_ascii=False)

    # Use apply_chat_template to automatically handle format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ví dụ:\n{FEW_SHOT}\n\nPhòng cần tư vấn:\n{user_msg}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    enc = tokenizer(prompt, return_tensors='pt')
    input_ids = enc['input_ids'].to(llm.device)
    attention_mask = enc['attention_mask'].to(llm.device)
    prompt_len = input_ids.shape[-1]
    
    layout = None
    
    def _extract_json(text):
        text = re.sub(r'^```(?:json)?\n?', '', text.strip())
        text = re.sub(r'\n?```$', '', text).strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return m.group(0) if m else text

    for attempt in range(max_retries + 1):
        temp = 0.25 + attempt * 0.15
        print(f'  Lần {attempt+1}/{max_retries+1} | temp={temp:.2f} | Đang sinh layout...')
        with torch.no_grad():
            out_ids = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0][prompt_len:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        try:
            layout = json.loads(_extract_json(raw_text))
            if 'furniture' not in layout or not isinstance(layout['furniture'], list):
                raise ValueError('Thiếu key "furniture" hoặc định dạng sai')
            print(f'  JSON OK - LLM đề xuất {len(layout["furniture"])} món đồ')
            break
        except (json.JSONDecodeError, ValueError) as e:
            print(f'  [Lỗi parse JSON] {e}')
            if attempt == max_retries:
                print('  -> Sử dụng fallback layout template do LLM lỗi sinh JSON.')
                layout = _fallback_layout(spec)
                
    # Clean up model from GPU memory immediately
    print("[LLM] Đang giải phóng VRAM của LLM...")
    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"VRAM giải phóng thành công. Hiện tại: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Post-process layout
    print('  [Post-process] Chuẩn hóa kích thước, bổ sung đồ bị thiếu và chạy tối ưu vị trí (Auto Layout)...')
    layout['furniture'] = normalize_furniture_dimensions(layout['furniture'])
    layout['furniture'] = ensure_must_have(layout['furniture'], spec)
    layout['furniture'] = auto_layout(layout['furniture'], W, L)
    return layout

def draw_2d_floor_plan(spec, layout):
    """
    Draw 2D floor plan using rotated polygons and save to output directory.
    """
    print("[2D Plan] Đang vẽ sơ đồ mặt bằng 2D...")
    COLOR_MAP = {
        'sofa':'#A8C4E0',         'ghe sofa':'#A8C4E0',
        'coffee table':'#C8DDB0', 'ban tra':'#C8DDB0',
        'tv stand':'#7A9E7E',     'ke tivi':'#7A9E7E',
        'armchair':'#B8D4E8',     'ghe banh':'#B8D4E8',
        'rug':'#E8DCC8',          'tham':'#E8DCC8',
        'floor lamp':'#F0C080',   'den san':'#F0C080',
        'bookshelf':'#D4A890',    'ke sach':'#D4A890',
        'side table':'#C0D4A8',   'ban phu':'#C0D4A8',
        'double bed':'#A8B8E0',   'giuong doi':'#A8B8E0',
        'wardrobe':'#C8A8D0',     'tu quan ao':'#C8A8D0',
        'nightstand':'#D0C8A8',   'tu dau giuong':'#D0C8A8',
        'desk':'#A8D0C8',         'ban lam viec':'#A8D0C8',
        'dining table':'#E0C898', 'ban an':'#E0C898',
        'dining chair':'#C8D4A8', 'ghe an':'#C8D4A8',
    }
    FB_COLORS = ['#B8C8D8','#C8D8B8','#D8C8B8','#C8B8D8','#D8D8B8','#B8D8C8','#D8B8C8']

    def _item_color(item, idx):
        nm = item.get('name_vi', item.get('name', '')).lower()
        for k, v in COLOR_MAP.items():
            if k in nm:
                return v
        return FB_COLORS[idx % len(FB_COLORS)]

    W, L = spec['width'], spec['length']
    fig, ax = plt.subplots(figsize=(9, max(6.0, 9.0 * L / W)))
    ax.set_facecolor('#FAFAF7')
    fig.patch.set_facecolor('#FFFFFF')

    # Draw room border
    ax.add_patch(mpatches.Rectangle(
        (0, 0), W, L, facecolor='#F5F2EC', edgecolor='#2C2C2A', linewidth=2.5, zorder=1
    ))
    # Draw grids
    for xg in np.arange(0.5, W, 0.5):
        ax.axvline(xg, color='#CCCCCC', lw=0.3, ls=':')
    for yg in np.arange(0.5, L, 0.5):
        ax.axhline(yg, color='#CCCCCC', lw=0.3, ls=':')

    for idx, item in enumerate(layout.get('furniture', [])):
        cx, cy = get_item_center(item)
        w,  ld = float(item.get('w', 0.5)), float(item.get('l', 0.5))
        rot    = float(item.get('rotation', 0))
        nm     = item.get('name_vi') or item.get('name', '')
        col    = _item_color(item, idx)
        
        # Calculate rotated corners and draw polygon
        corners = get_rotated_corners(cx, cy, w, ld, rot)
        ax.add_patch(mpatches.Polygon(
            corners, closed=True,
            facecolor=col, edgecolor='#444444', lw=1.0, alpha=0.88, zorder=2
        ))
        
        # Draw labels
        fs = min(8.0, w * 9, ld * 9)
        if fs >= 5.0 and w >= 0.3 and ld >= 0.25:
            ax.text(cx, cy, nm, ha='center', va='center', fontsize=fs,
                    color='#1A1A1A',
                    path_effects=[mpe.withStroke(linewidth=2, foreground='white')], zorder=3)
        if fs >= 6.0 and ld >= 0.4:
            ax.text(cx, cy - ld*0.22, f'{w:.1f}x{ld:.1f}m', ha='center', va='center',
                    fontsize=fs*0.7, color='#666660', zorder=3)

    # Room dimensions annotations
    ax.annotate('', xy=(W,-0.22), xytext=(0,-0.22),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=1.5))
    ax.text(W/2, -0.38, f'{W:.1f} m', ha='center', va='top', fontsize=9, color='#333')
    ax.annotate('', xy=(-0.22,L), xytext=(-0.22,0),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=1.5))
    ax.text(-0.45, L/2, f'{L:.1f} m', ha='right', va='center', fontsize=9, color='#333', rotation=90)
    ax.text(W-0.12, L-0.12, 'N', ha='right', va='top', fontsize=11, color='#185FA5', fontweight='bold')

    ax.set_xlim(-0.65, W+0.25)
    ax.set_ylim(-0.6, L+0.25)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(
        f"{spec['function'].title()}  .  {spec['style'].title()}\n"
        f"{W}m x {L}m  .  {spec['area_m2']:.1f} m2  .  Trần {spec['height']}m",
        fontsize=11, pad=12
    )
    plt.tight_layout()
    preview_path = OUTPUT_DIR / 'floor_plan_preview.png'
    plt.savefig(preview_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[2D Plan] Sơ đồ mặt bằng saved: {preview_path}')
    return preview_path

def main():
    # Setup test room spec
    parser = RoomSpecParser()
    room_spec = parser.parse_dict({
        'length'     : 6.0,
        'width'      : 4.5,
        'height'     : 2.7,
        'function'   : 'phong khach',
        'style'      : 'scandinavian',
        'windows'    : 2,
        'doors'      : 1,
        'orientation': 'S',
        'budget'     : 'mid',
        'notes'      : 'Cua so lon huong Nam',
    })
    
    print("=== KHỞI ĐỘNG HỆ THỐNG THIẾT KẾ NỘI THẤT (RoDe) ===")
    print(f"Thông số phòng đầu vào:\n{json.dumps(room_spec, ensure_ascii=False, indent=2)}")
    
    # 1. LLM consult layout
    layout = query_llm_layout(room_spec)
    
    # Save Layout JSON
    layout_path = OUTPUT_DIR / 'layout.json'
    with open(layout_path, 'w', encoding='utf-8') as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    print(f'\n[Layout] Bố cục nội thất đã lưu: {layout_path}')
    
    # 2. Draw 2D Floor Plan
    draw_2d_floor_plan(room_spec, layout)
    
    # 3. Export 3D CAD DXF file
    dxf_path = str(OUTPUT_DIR / 'room_layout.dxf')
    export_dxf(room_spec, layout, dxf_path)
    
    # 4. Generate Depth Map
    print("\n[Depth Map] Đang sinh bản đồ độ sâu...")
    depth_img = create_depth_map(room_spec, layout)
    depth_path = OUTPUT_DIR / 'depth_map.png'
    depth_img.save(depth_path)
    print(f'[Depth Map] Bản đồ độ sâu đã lưu: {depth_path}')
    
    # 5. Build prompt and Render using SDXL + ControlNet
    pos_prompt, neg_prompt = build_image_prompt(room_spec, layout)
    print(f'\n[Prompt] Positive: {pos_prompt[:180]}...')
    
    renderer = RoomRenderer()
    render_path, comp_path = renderer.render(depth_path, pos_prompt, neg_prompt)
    
    print("\n=== KẾT QUẢ ĐẦU RA (OUTPUT FILES) ===")
    for fp in sorted(OUTPUT_DIR.iterdir()):
        sz = fp.stat().st_size
        unit = "KB" if sz < 1e6 else "MB"
        val = sz / 1e3 if sz < 1e6 else sz / 1e6
        print(f'  - {fp.name:32s}  {val:6.1f} {unit}')
        
    print("\n=== HOÀN THÀNH PIPELINE CHẠY LOCAL THÀNH CÔNG ===")

if __name__ == "__main__":
    main()

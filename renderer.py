import gc
import re
import time
from pathlib import Path
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

from config import (
    SDXL_MODEL_ID, CONTROLNET_MODEL_ID, VAE_MODEL_ID,
    STYLE_PROMPTS, ROOM_EN, LIGHT_DIR, QUALITY, NEGATIVE, OUTPUT_DIR
)
from layout import get_item_center, get_rotated_corners

def create_depth_map(spec, layout, size=1024):
    img  = Image.new('RGB', (size, size), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    W, L = spec['width'], spec['length']
    sx, sy = size / W, size / L
    
    def px(x, y):  # flip Y to match top-down image coordinates
        return int(x * sx), int((L - y) * sy)

    def draw_rect(x1, y1, x2, y2, fill):
        p1, p2 = px(x1, y1), px(x2, y2)
        x_min = min(p1[0], p2[0])
        y_min = min(p1[1], p2[1])
        x_max = max(p1[0], p2[0])
        y_max = max(p1[1], p2[1])
        draw.rectangle([x_min, y_min, x_max, y_max], fill=fill)
        
    draw_rect(0, 0, W, L, fill=(60, 60, 60))
    t = 0.15
    for x1, y1, x2, y2 in [(0, 0, W, t), (0, L-t, W, L), (0, t, t, L-t), (W-t, t, W, L-t)]:
        draw_rect(x1, y1, x2, y2, fill=(120, 120, 120))
        
    max_h = max((float(it.get('h', 0.5)) for it in layout.get('furniture', [])), default=1.0)
    for it in layout.get('furniture', []):
        cx, cy = get_item_center(it)
        w, ld  = float(it.get('w', 0.5)), float(it.get('l', 0.5))
        h      = float(it.get('h', 0.4))
        rot    = float(it.get('rotation', 0))
        
        bright = int(150 + 100 * (h / max_h))
        c = (bright, bright, bright)
        
        # Calculate rotated corners and draw polygon
        corners = get_rotated_corners(cx, cy, w, ld, rot)
        pixel_corners = [px(rx, ry) for rx, ry in corners]
        draw.polygon(pixel_corners, fill=c, outline=(210, 210, 210))
        
    return img

def build_image_prompt(spec, layout):
    style   = STYLE_PROMPTS.get(spec['style'].lower(), f'{spec["style"]} interior')
    room    = ROOM_EN.get(spec['function'], spec['function'])
    light   = LIGHT_DIR.get(spec.get('orientation', 'N'), 'natural light')
    items   = ', '.join(it.get('name_vi') or it.get('name', '') for it in layout.get('furniture', [])[:6])
    mats    = ', '.join(layout.get('materials', [])[:3])
    kw      = layout.get('image_prompt_keywords', '')
    
    parts   = [p for p in [style, room, items, mats, light, kw, QUALITY] if p.strip()]
    prompt  = ', '.join(p.strip(', ') for p in parts)
    return re.sub(r',\s*,', ',', prompt), NEGATIVE

class RoomRenderer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def render(self, depth_path, pos_prompt, neg_prompt, seed=42):
        if self.device != "cuda":
            print("WARNING: GPU is not available. Running on CPU will be extremely slow.")
            
        print('Loading ControlNet...')
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.float16, use_safetensors=True,
        )
        print('Loading VAE...')
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=torch.float16)
        
        print(f'Loading SDXL: {SDXL_MODEL_ID}...')
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL_ID,
            controlnet=controlnet, vae=vae,
            torch_dtype=torch.float16, use_safetensors=True,
        ).to(self.device)
        
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print('xformers enabled.')
        except Exception:
            print('xformers is not available - using attention slicing.')
            
        print(f'Pipeline ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')
        
        cond_img = load_image(str(depth_path)).resize((1024, 1024))
        
        GEN_CFG = dict(
            prompt=pos_prompt, negative_prompt=neg_prompt,
            image=cond_img, controlnet_conditioning_scale=0.65,
            num_inference_steps=30, guidance_scale=7.5,
            width=1024, height=1024,
            generator=torch.Generator(self.device).manual_seed(seed),
        )
        
        print('Generating image...')
        t0 = time.time()
        render = pipe(**GEN_CFG).images[0]
        elapsed = time.time() - t0
        
        # Save images
        img_path = OUTPUT_DIR / 'room_render.png'
        render.save(img_path)
        print(f'Render completed in {elapsed:.0f}s -> {img_path}')
        
        # Save comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(cond_img); axes[0].set_title('Depth map'); axes[0].axis('off')
        axes[1].imshow(render); axes[1].set_title(f"Rendered ({elapsed:.0f}s)"); axes[1].axis('off')
        plt.tight_layout()
        comparison_path = OUTPUT_DIR / 'comparison.png'
        plt.savefig(comparison_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        # Clean up VRAM immediately
        print("Cleaning up rendering VRAM...")
        del pipe, controlnet, vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return img_path, comparison_path

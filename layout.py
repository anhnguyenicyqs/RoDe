import math
from config import WALL_GAP, ITEM_GAP, ROOM_TEMPLATES, FUNC_ALIAS, MUST_HAVE

def get_effective_dims(item):
    w = float(item.get('w', 0.5))
    l = float(item.get('l', 0.5))
    rot = float(item.get('rotation', 0))
    if round(rot) % 180 == 90:
        return l, w
    return w, l

def get_item_center(item):
    x = float(item.get('x', 0))
    y = float(item.get('y', 0))
    w = float(item.get('w', 0.5))
    l = float(item.get('l', 0.5))
    return x + w / 2.0, y + l / 2.0

def set_item_center(item, cx, cy):
    w = float(item.get('w', 0.5))
    l = float(item.get('l', 0.5))
    item['x'] = round(cx - w / 2.0, 2)
    item['y'] = round(cy - l / 2.0, 2)

def get_rotated_corners(cx, cy, w, l, rot_deg):
    rad = math.radians(rot_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    local_corners = [
        (-w / 2.0, -l / 2.0),
        (w / 2.0, -l / 2.0),
        (w / 2.0, l / 2.0),
        (-w / 2.0, l / 2.0),
    ]
    rotated = []
    for lx, ly in local_corners:
        rx = cx + lx * cos_r - ly * sin_r
        ry = cy + lx * sin_r + ly * cos_r
        rotated.append((rx, ry))
    return rotated

def _overlaps(a, b, gap=ITEM_GAP):
    name_a = (a.get('name_vi') or a.get('name', '')).lower()
    name_b = (b.get('name_vi') or b.get('name', '')).lower()
    exceptions = ['rug', 'tham', 'den san', 'floor lamp', 'ceiling lamp', 'den tran', 'den chum', 'painting', 'tranh']
    if any(ex in name_a for ex in exceptions) or any(ex in name_b for ex in exceptions):
        return False
        
    aw, al = get_effective_dims(a)
    bw, bl = get_effective_dims(b)
    acx, acy = get_item_center(a)
    bcx, bcy = get_item_center(b)
    
    ax_min, ax_max = acx - aw / 2.0, acx + aw / 2.0
    ay_min, ay_max = acy - al / 2.0, acy + al / 2.0
    bx_min, bx_max = bcx - bw / 2.0, bcx + bw / 2.0
    by_min, by_max = bcy - bl / 2.0, bcy + bl / 2.0
    
    return (
        ax_min - gap < bx_max and
        ax_max + gap > bx_min and
        ay_min - gap < by_max and
        ay_max + gap > by_min
    )

def _clamp(item, W, L):
    c = dict(item)
    ew, el = get_effective_dims(c)
    cx, cy = get_item_center(c)
    cx = max(WALL_GAP + ew / 2.0, min(cx, W - WALL_GAP - ew / 2.0))
    cy = max(WALL_GAP + el / 2.0, min(cy, L - WALL_GAP - el / 2.0))
    set_item_center(c, cx, cy)
    return c

def _find_free_spiral(item, placed, W, L):
    ew, el = get_effective_dims(item)
    orig_cx, orig_cy = get_item_center(item)
    step = 0.1
    x_steps = int(W / step)
    y_steps = int(L / step)
    candidates = []
    for j in range(y_steps + 1):
        cy = round(j * step, 2)
        if cy < WALL_GAP + el / 2.0 or cy > L - WALL_GAP - el / 2.0:
            continue
        for i in range(x_steps + 1):
            cx = round(i * step, 2)
            if cx < WALL_GAP + ew / 2.0 or cx > W - WALL_GAP - ew / 2.0:
                continue
            dist = math.hypot(cx - orig_cx, cy - orig_cy)
            candidates.append((dist, cx, cy))
            
    candidates.sort(key=lambda x: x[0])
    for dist, cx, cy in candidates:
        cand = dict(item)
        set_item_center(cand, cx, cy)
        if not any(_overlaps(cand, p) for p in placed):
            return cand
    return None

def auto_layout(furniture, W, L):
    placed = []
    for item in furniture:
        item = _clamp(item, W, L)
        if any(_overlaps(item, p) for p in placed):
            fixed = _find_free_spiral(item, placed, W, L)
            if fixed is None:
                n = item.get('name_vi', item.get('name', '?'))
                print(f'    [skip] het cho: {n}')
                continue
            item = fixed
        placed.append(item)
    return placed

def _get_template_key(spec):
    func = spec['function'].lower()
    if func in ROOM_TEMPLATES:
        return func
    for alias, key in FUNC_ALIAS.items():
        if alias in func or func in alias:
            return key
    return 'phong khach'  

def ensure_must_have(furniture, spec):
    tkey     = _get_template_key(spec)
    template = ROOM_TEMPLATES.get(tkey, [])
    existing = {it['name'].lower() for it in furniture}
    for must in MUST_HAVE.get(tkey, []):
        if not any(must in n for n in existing):
            tmpl = next((t for t in template if must in t['name']), None)
            if tmpl:
                added = dict(tmpl)
                added.update({'x': 0.0, 'y': 0.0, 'rotation': 0,
                               'material': 'wood', 'color': 'natural'})
                print(f"    [bo sung] '{tmpl['name_vi']}' bi LLM bo sot")
                furniture.insert(0, added)
    return furniture

def _fallback_layout(spec):
    tkey  = _get_template_key(spec)
    items = ROOM_TEMPLATES.get(tkey, ROOM_TEMPLATES['phong khach'])
    return {
        'furniture': [
            dict(t, x=0.0, y=0.0, rotation=0, material='wood', color='natural')
            for t in items
        ],
        'style_description': f'{spec["style"]} interior design',
        'color_palette': ['#F5F0E8', '#D4C5B0', '#8B7355'],
        'materials': ['wood', 'fabric'],
        'lighting': 'Natural daylight',
        'image_prompt_keywords': f'{spec["style"]} {spec["function"]}',
    }

def normalize_furniture_dimensions(furniture):
    normalized = []
    for item in furniture:
        it = dict(item)
        
        # 1. Fix height key if the LLM output "o" instead of "h"
        if "o" in it and "h" not in it:
            it["h"] = it.pop("o")
        if "h" not in it:
            it["h"] = 0.5  # default height
            
        # Ensure numeric values
        for k in ("w", "l", "h", "x", "y", "rotation"):
            if k in it:
                try:
                    it[k] = float(it[k])
                except (ValueError, TypeError):
                    it[k] = 0.0
            else:
                it[k] = 0.0
                
        # 2. Fix typos in name
        name = str(it.get("name", "")).lower().strip()
        if name.endswith("tableable"):
            name = name.replace("tableable", "table")
        elif name.endswith("tabletable"):
            name = name.replace("tabletable", "table")
        it["name"] = name
        
        # 3. Clean Vietnamese translation
        name_vi = str(it.get("name_vi", "")).lower().strip()
        if "ban tra" in name_vi or "coffee" in name:
            it["name_vi"] = "ban tra"
        elif "ghe sofa" in name_vi or "sofa" in name:
            it["name_vi"] = "ghe sofa"
        elif "ke tivi" in name_vi or "tv" in name:
            it["name_vi"] = "ke tivi"
        elif "ghe banh" in name_vi or "ghe banan" in name_vi or "armchair" in name:
            it["name_vi"] = "ghe banh"
            
        # 4. Enforce realistic minimum dimensions
        if "sofa" in name:
            it["w"] = max(it["w"], 1.8)
            it["l"] = max(it["l"], 0.8)
            it["h"] = max(it["h"], 0.7)
        elif "coffee" in name or "ban tra" in name_vi:
            it["w"] = max(it["w"], 0.9)
            it["l"] = max(it["l"], 0.5)
            it["h"] = max(it["h"], 0.4)
        elif "tv" in name or "ke tivi" in name_vi:
            it["w"] = max(it["w"], 1.4)
            it["l"] = max(it["l"], 0.4)
            it["h"] = max(it["h"], 0.4)
        elif "armchair" in name or "ghe banh" in name_vi:
            it["w"] = max(it["w"], 0.7)
            it["l"] = max(it["l"], 0.7)
            it["h"] = max(it["h"], 0.7)
            
        normalized.append(it)
    return normalized


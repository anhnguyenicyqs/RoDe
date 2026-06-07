import os
from pathlib import Path

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SDXL_MODEL_ID = "SG161222/RealVisXL_V4.0"
CONTROLNET_MODEL_ID = "diffusers/controlnet-depth-sdxl-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

# Layout parameters
WALL_GAP = 0.05
ITEM_GAP = 0.06

# Room design templates and aliases
ROOM_TEMPLATES = {
    'phong khach': [
        {'name': 'sofa',         'name_vi': 'ghe sofa',       'w': 2.2,  'l': 0.9,  'h': 0.85},
        {'name': 'coffee table', 'name_vi': 'ban tra',         'w': 1.1,  'l': 0.6,  'h': 0.45},
        {'name': 'tv stand',     'name_vi': 'ke tivi',         'w': 1.6,  'l': 0.4,  'h': 0.5 },
        {'name': 'armchair',     'name_vi': 'ghe banh',        'w': 0.8,  'l': 0.8,  'h': 0.85},
        {'name': 'armchair',     'name_vi': 'ghe banh',        'w': 0.8,  'l': 0.8,  'h': 0.85},
        {'name': 'side table',   'name_vi': 'ban phu',         'w': 0.5,  'l': 0.5,  'h': 0.55},
        {'name': 'floor lamp',   'name_vi': 'den san',         'w': 0.3,  'l': 0.3,  'h': 1.6 },
        {'name': 'rug',          'name_vi': 'tham',            'w': 2.0,  'l': 1.4,  'h': 0.01},
    ],
    'phong ngu': [
        {'name': 'double bed',   'name_vi': 'giuong doi',      'w': 1.6,  'l': 2.0,  'h': 0.5 },
        {'name': 'nightstand',   'name_vi': 'tu dau giuong',   'w': 0.5,  'l': 0.45, 'h': 0.55},
        {'name': 'nightstand',   'name_vi': 'tu dau giuong',   'w': 0.5,  'l': 0.45, 'h': 0.55},
        {'name': 'wardrobe',     'name_vi': 'tu quan ao',      'w': 1.8,  'l': 0.6,  'h': 2.0 },
        {'name': 'dresser',      'name_vi': 'tu ngan keo',     'w': 1.0,  'l': 0.45, 'h': 0.85},
        {'name': 'lounge chair', 'name_vi': 'ghe thu gian',    'w': 0.75, 'l': 0.8,  'h': 0.8 },
    ],
    'phong lam viec': [
        {'name': 'desk',         'name_vi': 'ban lam viec',    'w': 1.4,  'l': 0.7,  'h': 0.75},
        {'name': 'office chair', 'name_vi': 'ghe van phong',   'w': 0.65, 'l': 0.65, 'h': 1.1 },
        {'name': 'bookshelf',    'name_vi': 'ke sach',         'w': 0.8,  'l': 0.3,  'h': 2.0 },
        {'name': 'bookshelf',    'name_vi': 'ke sach',         'w': 0.8,  'l': 0.3,  'h': 2.0 },
        {'name': 'lounge chair', 'name_vi': 'ghe phu',         'w': 0.75, 'l': 0.8,  'h': 0.8 },
    ],
    'phong an': [
        {'name': 'dining table', 'name_vi': 'ban an',          'w': 1.6,  'l': 0.9,  'h': 0.75},
        {'name': 'dining chair', 'name_vi': 'ghe an',          'w': 0.45, 'l': 0.45, 'h': 0.9 },
        {'name': 'dining chair', 'name_vi': 'ghe an',          'w': 0.45, 'l': 0.45, 'h': 0.9 },
        {'name': 'dining chair', 'name_vi': 'ghe an',          'w': 0.45, 'l': 0.45, 'h': 0.9 },
        {'name': 'dining chair', 'name_vi': 'ghe an',          'w': 0.45, 'l': 0.45, 'h': 0.9 },
        {'name': 'sideboard',    'name_vi': 'tu bep thap',     'w': 1.4,  'l': 0.45, 'h': 0.85},
    ],
}

FUNC_ALIAS = {
    'phong khach'   : 'phong khach',
    'living room'   : 'phong khach',
    'phong ngu'     : 'phong ngu',
    'bedroom'       : 'phong ngu',
    'phong lam viec': 'phong lam viec',
    'home office'   : 'phong lam viec',
    'phong an'      : 'phong an',
    'dining room'   : 'phong an',
}

MUST_HAVE = {
    'phong khach'   : ['sofa'],
    'phong ngu'     : ['double bed'],
    'phong lam viec': ['desk', 'office chair'],
    'phong an'      : ['dining table'],
}

# LLM Prompts
SYSTEM_PROMPT = (
    'Ban la chuyen gia thiet ke noi that. Nhiem vu cua ban la bo tri mat bang noi that hop ly va tham my.\n'
    'Tra ve JSON hop le DUY NHAT, khong co text nao khac.\n\n'
    'QUY TAC KICH THUOC NOI THAT (BAT BUOC TUAN THU):\n'
    '- Sofa (ghe sofa): w=1.8 - 2.4, l=0.8 - 1.0, h=0.7 - 0.9\n'
    '- Coffee table (ban tra): w=0.9 - 1.2, l=0.5 - 0.7, h=0.4 - 0.5\n'
    '- TV stand (ke tivi): w=1.4 - 2.0, l=0.4 - 0.5, h=0.4 - 0.6\n'
    '- Armchair (ghe banh): w=0.7 - 0.9, l=0.7 - 0.9, h=0.7 - 0.9\n'
    '- Double bed (giuong doi): w=1.6 - 1.8, l=2.0 - 2.2, h=0.4 - 1.0\n'
    '- Wardrobe (tu quan ao): w=1.5 - 2.0, l=0.6 - 0.7, h=2.0 - 2.2\n'
    '- Nightstand (tu dau giuong): w=0.4 - 0.5, l=0.4 - 0.5, h=0.5 - 0.6\n'
    '- Desk (ban lam viec): w=1.2 - 1.5, l=0.6 - 0.7, h=0.75\n\n'
    'QUY TAC BO CUC PHONG (QUAN TRONG):\n'
    '1. Sofa phai ke sat mot buc tuong (vi du: tuong duoi y=0.1, hoac tuong trai x=0.1).\n'
    '2. Ban tra (coffee table) luon di kem va dat song song phia truoc sofa (cach sofa khoang 0.3m - 0.5m).\n'
    '3. Ke tivi (tv stand) phai dat o buc tuong doi dien sofa va song song voi sofa de nguoi ngoi xem duoc tivi.\n'
    '4. Ghe banh (armchair) dat o hai ben ban tra, huong ve phia tivi hoac sofa tao thanh goc quay quan sat tot.\n'
    '5. Giuong doi (double bed) phai dat dau giuong sat vao tuong sau (y=length-l-0.1) hoac tuong ben. Hai ben dau giuong la 2 tu dau giuong (nightstand).\n'
    '6. Tu quan ao (wardrobe) ke sat doc theo mot tuong con lai, khong chan loi di hoac cua so.\n\n'
    'QUY TAC TOA DO:\n'
    '- Goc (0,0) la goc duoi-trai phong. x tang sang phai (max=width), y tang len tren (max=length).\n'
    '- Do sat tuong truoc: y=0.1 | sat tuong sau: y=length-l-0.1\n'
    '- KHONG de do chong nhau. KHONG de do vuot tuong.\n\n'
    'JSON SCHEMA (QUAN TRONG: dung dung cac key "w", "l", "h" cho kich thuoc, "x", "y" cho toa do, "rotation" cho goc quay (0, 90, 180, 270)):\n'
    '{"furniture":[{"name":"sofa","name_vi":"ghe sofa","w":2.2,"l":0.9,"h":0.85,"x":1.05,"y":0.1,"rotation":0,"material":"linen","color":"light gray"}],'
    '"style_description":"...","color_palette":["#hex"],"materials":["..."],"lighting":"...","image_prompt_keywords":"..."}'
)

FEW_SHOT = (
    'INPUT: {"length":6.0,"width":4.5,"height":2.7,"function":"phong khach",'
    '"style":"scandinavian","windows":2,"doors":1,"budget":"mid"}\n\n'
    'OUTPUT: {"furniture":['
    '{"name":"sofa","name_vi":"ghe sofa","w":2.2,"l":0.9,"h":0.85,"x":1.05,"y":0.1,"rotation":0,"material":"linen","color":"light gray"},'
    '{"name":"coffee table","name_vi":"ban tra","w":1.1,"l":0.6,"h":0.45,"x":1.35,"y":1.15,"rotation":0,"material":"oak","color":"natural oak"},'
    '{"name":"tv stand","name_vi":"ke tivi","w":1.6,"l":0.4,"h":0.5,"x":1.45,"y":5.5,"rotation":0,"material":"oak","color":"white oak"},'
    '{"name":"armchair","name_vi":"ghe banh","w":0.8,"l":0.8,"h":0.85,"x":0.1,"y":1.0,"rotation":0,"material":"wool","color":"beige"},'
    '{"name":"armchair","name_vi":"ghe banh","w":0.8,"l":0.8,"h":0.85,"x":3.6,"y":1.0,"rotation":0,"material":"wool","color":"beige"},'
    '{"name":"side table","name_vi":"ban phu","w":0.45,"l":0.45,"h":0.55,"x":0.1,"y":2.0,"rotation":0,"material":"oak","color":"natural"},'
    '{"name":"floor lamp","name_vi":"den san","w":0.3,"l":0.3,"h":1.6,"x":3.85,"y":2.1,"rotation":0,"material":"metal","color":"matte black"},'
    '{"name":"rug","name_vi":"tham","w":2.0,"l":1.4,"h":0.01,"x":1.1,"y":0.9,"rotation":0,"material":"wool","color":"cream"}'
    '],\n'
    '"style_description":"Scandinavian toi gian, go sang, vai linen, tong mau trung tinh.",\n'
    '"color_palette":["#F5F0E8","#D4C5B0","#8B7355","#3D3B38"],\n'
    '"materials":["linen","oak wood","wool"],\n'
    '"lighting":"Den tran 3000K + den san goc phong",\n'
    '"image_prompt_keywords":"scandinavian living room, linen sofa, oak coffee table, white walls, hygge, natural light"}'
)

# Rendering settings
STYLE_PROMPTS = {
    'scandinavian'      : 'Scandinavian interior, hygge, white walls, natural light wood, clean lines',
    'japandi'           : 'Japandi interior, wabi-sabi, earth tones, zen minimalism, natural materials',
    'modern'            : 'modern contemporary interior, sleek surfaces, neutral palette, statement lighting',
    'minimalist'        : 'minimalist interior, sparse furniture, monochrome, negative space',
    'industrial'        : 'industrial loft, exposed brick, metal accents, Edison bulbs, raw materials',
    'bohemian'          : 'bohemian interior, layered textiles, indoor plants, warm earth tones',
    'mid-century modern': 'mid-century modern, walnut wood, mustard tones, geometric patterns',
    'wabi-sabi'         : 'wabi-sabi interior, imperfect textures, aged wood, muted palette',
    'classic'           : 'classic elegant interior, crown molding, symmetry, rich fabrics',
    'tropical'          : 'tropical interior, rattan furniture, palm motifs, airy space',
}

ROOM_EN = {
    'phong khach'    : 'living room',
    'living room'    : 'living room',
    'phong ngu'      : 'bedroom',
    'bedroom'        : 'bedroom',
    'phong lam viec' : 'home office',
    'home office'    : 'home office',
    'phong an'       : 'dining room',
    'dining room'    : 'dining room',
}

LIGHT_DIR = {
    'S' : 'warm southern sunlight',
    'N' : 'soft northern light',
    'E' : 'golden morning light',
    'W' : 'warm afternoon light'
}

QUALITY = 'professional interior photography, 8k, photorealistic, natural daylight, architectural visualization'
NEGATIVE = 'cartoon, painting, sketch, anime, blurry, low quality, deformed, ugly, text, watermark'

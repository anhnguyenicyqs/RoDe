import math
from pathlib import Path
import ezdxf
from ezdxf.enums import TextEntityAlignment

# Define layers color codes (dxf color indices)
LAYER_COLORS = {
    'WALL': 8, 'FLOOR': 9, 'CEILING': 254,
    'BED': 4, 'SOFA': 5, 'TABLE': 2, 'CHAIR': 3,
    'WARDROBE': 6, 'CABINET': 7, 'DESK': 30,
    'LAMP': 40, 'RUG': 50, 'SHELF': 60, 'DEFAULT': 1,
}

def _layer_name(name):
    u = name.upper().replace(' ', '_').replace('-', '_')
    for k in LAYER_COLORS:
        if k in u:
            return k
    return u

def _ensure_layer(doc, name, color=None):
    c = color if color is not None else LAYER_COLORS.get(name, LAYER_COLORS['DEFAULT'])
    if name not in doc.layers:
        doc.layers.add(name, dxfattribs={'color': c})
    else:
        doc.layers.get(name).dxf.color = c

def _add_box(msp, x, y, z, w, l, h, layer, rot_deg=0.0):
    """6-face solid box using 3DFACE entities."""
    pts = [
        (x,   y,   z),   (x+w, y,   z),   (x+w, y+l, z),   (x,   y+l, z),
        (x,   y,   z+h), (x+w, y,   z+h), (x+w, y+l, z+h), (x,   y+l, z+h),
    ]
    if rot_deg % 360 != 0:
        cx, cy = x + w/2, y + l/2
        rad = math.radians(rot_deg)
        cr, sr = math.cos(rad), math.sin(rad)
        def _r(px, py):
            dx, dy = px-cx, py-cy
            return cx + dx*cr - dy*sr, cy + dx*sr + dy*cr
        pts = [(_r(p[0],p[1])[0], _r(p[0],p[1])[1], p[2]) for p in pts]
    a = {'layer': layer}
    for face in [
        [pts[0],pts[1],pts[2],pts[3]],
        [pts[4],pts[5],pts[6],pts[7]],
        [pts[0],pts[1],pts[5],pts[4]],
        [pts[3],pts[2],pts[6],pts[7]],
        [pts[0],pts[3],pts[7],pts[4]],
        [pts[1],pts[2],pts[6],pts[5]],
    ]:
        msp.add_3dface(face, dxfattribs=a)

def _add_walls(doc, msp, W, L, H, t=15.0):
    for ln in ('WALL','WALL_FRONT','WALL_BACK','WALL_LEFT','WALL_RIGHT','FLOOR','CEILING'):
        _ensure_layer(doc, ln, LAYER_COLORS.get(ln.split('_')[0], LAYER_COLORS['WALL']))
    for wx,wy,ww,wl,wlyr in [
        (0,   0,   W,   t,   'WALL_FRONT'),
        (0,   L-t, W,   t,   'WALL_BACK'),
        (0,   t,   t,   L-2*t,'WALL_LEFT'),
        (W-t, t,   t,   L-2*t,'WALL_RIGHT'),
    ]:
        _add_box(msp, wx, wy, 0, ww, wl, H, wlyr)
    msp.add_3dface([(0,0,0),(W,0,0),(W,L,0),(0,L,0)], dxfattribs={'layer':'FLOOR'})
    
    # Ceiling: 2D lwpolyline with elevation
    msp.add_lwpolyline(
        [(0,0),(W,0),(W,L),(0,L)], close=True,
        dxfattribs={'layer':'CEILING','elevation':H}
    )

def export_dxf(spec, layout, output_path):
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 5  # cm (5 represents Centimeters in DXF header)
    msp = doc.modelspace()
    W = spec['width']  * 100
    L = spec['length'] * 100
    H = spec['height'] * 100
    
    for ln, ci in LAYER_COLORS.items():
        _ensure_layer(doc, ln, ci)
        
    _add_walls(doc, msp, W, L, H)
    
    msp.add_lwpolyline(
        [(0,0),(W,0),(W,L),(0,L)], close=True,
        dxfattribs={'layer':'FLOOR','lineweight':50}
    )
    
    for item in layout.get('furniture', []):
        ix = float(item.get('x', 0)) * 100
        iy = float(item.get('y', 0)) * 100
        iw = max(float(item.get('w', 0.5)), 0.1) * 100
        il = max(float(item.get('l', 0.5)), 0.1) * 100
        ih = max(float(item.get('h', 0.4)), 0.05) * 100
        rot = float(item.get('rotation', 0))
        lyr = _layer_name(item.get('name', 'item'))
        
        _ensure_layer(doc, lyr, LAYER_COLORS.get(lyr, LAYER_COLORS['DEFAULT']))
        _add_box(msp, ix, iy, 0, iw, il, ih, lyr, rot_deg=rot)
        
        label  = (item.get('name_vi') or item.get('name', ''))[:14]
        txt_h  = max(5.0, min(iw, il) * 0.22)
        te = msp.add_text(label, dxfattribs={'layer': lyr, 'height': txt_h})
        te.set_placement((ix + iw/2, iy + il/2), align=TextEntityAlignment.MIDDLE_CENTER)
        
    doc.saveas(output_path)
    kb = Path(output_path).stat().st_size / 1024
    print(f'DXF saved: {output_path}  ({kb:.1f} KB)')
    return output_path

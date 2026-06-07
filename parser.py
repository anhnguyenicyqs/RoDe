import json
from pathlib import Path

VALID_ORIENTATIONS = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']

class RoomSpecParser:
    """
    Parse room spec tu file .json / .txt hoac dict inline.
    Truong bat buoc: length, width, height, function, style.
    """
    REQUIRED = {'length', 'width', 'height', 'function', 'style'}
    DEFAULTS  = {'windows': 1, 'doors': 1, 'orientation': 'N', 'budget': 'mid', 'notes': ''}

    def parse_file(self, filepath):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f'Khong tim thay: {filepath}')
        if path.suffix.lower() == '.json':
            with open(path, encoding='utf-8') as f:
                raw = json.load(f)
        elif path.suffix.lower() in ('.txt', '.text'):
            raw = {}
            with open(path, encoding='utf-8') as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#') or ':' not in line:
                        continue
                    k, _, v = line.partition(':')
                    raw[k.strip().lower()] = v.strip()
        else:
            raise ValueError(f'Chi ho tro .json / .txt')
        return self._normalize(raw)

    def parse_dict(self, data):
        return self._normalize(data)

    def _normalize(self, raw):
        raw = {k.lower().strip(): v for k, v in raw.items()}
        missing = self.REQUIRED - raw.keys()
        if missing:
            raise ValueError(f'Thieu truong bat buoc: {missing}')
        spec = dict(self.DEFAULTS)
        spec.update(raw)
        for field in ('length', 'width', 'height'):
            spec[field] = float(spec[field])
            if spec[field] <= 0:
                raise ValueError(f'`{field}` phai > 0')
        for field in ('windows', 'doors'):
            spec[field] = int(spec[field])
        if not (1.5 <= spec['length'] <= 50):
            raise ValueError(f'length phai 1.5-50m')
        if not (1.5 <= spec['width'] <= 50):
            raise ValueError(f'width phai 1.5-50m')
        if not (2.0 <= spec['height'] <= 6.0):
            raise ValueError(f'height phai 2-6m')
        spec['function']    = str(spec['function']).strip().lower()
        spec['style']       = str(spec['style']).strip().lower()
        spec['budget']      = str(spec['budget']).strip().lower()
        spec['orientation'] = str(spec['orientation']).strip().upper()
        if spec['orientation'] not in VALID_ORIENTATIONS:
            spec['orientation'] = 'N'
        spec['area_m2'] = round(spec['length'] * spec['width'], 2)
        return spec

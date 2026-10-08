"""Uniform k-trit formats. Intermediate BPW comes from scale/group overhead."""
import re

def parse_format(name, group=128):
    m = re.fullmatch(r'TKinteger-k(1[0-6]|[1-9])(?:-g([1-9][0-9]*)-s(12|16|32))?', name)
    if not m:
        raise ValueError(f'Unsupported executable format: {name}')
    k = int(m[1]); g = int(m[2]) if m[2] else group
    if not isinstance(g,int) or isinstance(g,bool) or not 1 <= g <= 65536:
        raise ValueError('Group size must be an integer from 1 to 65536')
    return dict(name=name, k=k, group=g, scale_bits=int(m[3]) if m[2] else (16 if k <= 5 else 32))

def storage(rows, cols, spec):
    g, k = spec['group'], spec['k']; groups = rows*((cols+g-1)//g)
    payload = groups*((g*k+4)//5) if spec['scale_bits']==12 else rows*((cols*k+4)//5)
    scales = (groups*spec['scale_bits']+7)//8
    return dict(payload=payload, scales=scales, total=payload+scales)

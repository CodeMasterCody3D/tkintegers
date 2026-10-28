"""Path-independent adapter of r2's size-preserving matrix writer.

The template is one quality9 layout shard with bare codes/scales/meta keys,
not a desktop manifest shard with tensor-name prefixes. This adapter preserves
the measured reshaping and value checks; its packaging has static checks only.
"""
from pathlib import Path
import hashlib
import struct

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save


def _header(path):
    with Path(path).open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        return handle.read(size)


def save_matrix_like(payload, template, destination):
    """Save to a new path, retaining template shapes/dtypes/header/file length.

    payload is the (signed int8 codes, int16 scales, integer meta) returned by
    refine(). A failed check leaves its candidate file for diagnosis; never use
    that file as an accepted artifact. No model layout or backup is written here.
    """
    template, destination = Path(template), Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite a matrix: ' + str(destination))
    with safe_open(str(template), framework='pt', device='cpu') as handle:
        shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
        dtypes = {key: handle.get_slice(key).get_dtype() for key in handle.keys()}
    if set(shapes) != {'codes', 'scales', 'meta'}:
        raise ValueError('Expected a quality9 matrix shard with codes/scales/meta')
    codes, scale_codes, meta = payload
    if codes.ndim != 2 or codes.shape[1] % 128 or codes.dtype != torch.int8:
        raise ValueError('Recipe v1 requires signed K5 matrix codes and group128')
    if bool(((codes < -121) | (codes > 121)).any()):
        raise ValueError('Illegal K5 code')
    if scale_codes.dtype != torch.int16 or meta.reshape(-1)[2:].tolist() != [10, 1]:
        raise ValueError('Recipe v1 requires integer-k10-log scales')
    raw = (payload[0].to(torch.int16) + 121).to(torch.uint8).reshape(shapes['codes']).cpu().contiguous()
    scales = payload[1].reshape(shapes['scales']).cpu().contiguous()
    metadata = payload[2].reshape(shapes['meta']).cpu().contiguous()
    # Same safetensors encoding; exclusive creation also prevents concurrent
    # callers from replacing each other's output between existence check/write.
    data = save({'codes': raw, 'scales': scales, 'meta': metadata})
    with destination.open('xb') as handle:
        handle.write(data)
    if destination.stat().st_size != template.stat().st_size:
        raise RuntimeError('K5 serialized matrix size changed')
    with safe_open(str(destination), framework='pt', device='cpu') as handle:
        if ({key: handle.get_slice(key).get_shape() for key in handle.keys()} != shapes or
                {key: handle.get_slice(key).get_dtype() for key in handle.keys()} != dtypes):
            raise RuntimeError('K5 serialized shape or dtype changed')
    if _header(destination) != _header(template):
        raise RuntimeError('K5 serialized header changed')
    check = load_file(str(destination), device=str(codes.device))
    if not torch.equal(check['codes'].reshape_as(payload[0]).to(torch.int16) - 121, payload[0].to(torch.int16)):
        raise RuntimeError('Saved K5 codes changed')
    if (not torch.equal(check['scales'].reshape_as(payload[1]), payload[1]) or
            not torch.equal(check['meta'].cpu().reshape_as(payload[2]), payload[2].cpu())):
        raise RuntimeError('Saved integer scales changed')
    return {'file': str(destination), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
            'template': str(template), 'exact_header_shape_dtype_size': True}

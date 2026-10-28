"""Reusable building blocks for the preserved AWQ/QEP/DAPTQ K5 recipe.

Numerical use requires the recorded CUDA environment and caller-owned carried
calibration. Importing this module does not start conversion, download, or upload.
The desktop converter does not invoke this method automatically.
"""
from pathlib import Path
import hashlib
import json


def load_recipe():
    """Return the recorded recipe without changing any execution settings."""
    return json.loads(Path(__file__).with_name('recipe.json').read_text())


def __getattr__(name):
    # Metadata/package verification needs no Torch installation. Numerical entry
    # points use the existing codec only after checking its measured identity.
    math_names = {'statistics', 'qep_target', 'refine', 'decode', 'energy', 'algebra_check'}
    if name not in math_names | {'fold_awq', 'save_matrix_like'}:
        raise AttributeError(name)
    import importlib.util
    spec = importlib.util.find_spec('tk_codec')
    expected = load_recipe()['dependencies']['tk_codec_sha256']
    if spec is None or spec.origin is None or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Recipe v1 requires its recorded tk_codec.py; run verify.py from the TKintegers checkout')
    if name in math_names:
        from .reference import quality9_k5_daptq
        return getattr(quality9_k5_daptq, name)
    if name == 'fold_awq':
        from .folding import fold_awq
        return fold_awq
    from .storage import save_matrix_like
    return save_matrix_like

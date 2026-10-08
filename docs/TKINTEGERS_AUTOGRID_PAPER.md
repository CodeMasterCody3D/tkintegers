# Tk-Integers and Autogrid

## A format-density designer for packed k-trit model storage

**Status:** engineering paper draft. The UI and accounting are implemented;
model-quality claims and broad novelty claims remain open until matched model
runs and primary-source comparisons are attached.

## Abstract

Tk-Integers represents a value with packed balanced base-3 trits and an
explicit scale stream. The number of trits per value (`k`) controls the code
resolution. The number of weights sharing a scale controls a second,
independent density axis: using more scales per weight increases bits per
weight without changing `k`. This paper introduces the Tk-Integer Designer,
built into the MIT-licensed Autogrid desktop application. It lets a user build
arbitrary mixtures of k-trit planes, scale densities, scale precisions, and
optional carried-residual planes, then reports both ideal and finite byte
payload density.

The important distinction is between a code-depth ladder and a scale-density
ladder. A single k choice does not identify a complete format. For example,
the remembered S12 family is k1 with a 12-bit log scale, but `g20` and `g15`
are different formats: they are 2.2 and 2.4 ideal bits/weight respectively.

## 1. Representation

Five balanced trits are packed into one byte. A k-trit code therefore costs
`1.6*k` ideal bits per value. If one scale is shared by `g` weights and the
scale has `s` bits, the ideal density of one plane is:

```text
bpw_ideal = 1.6*k + s/g
```

The finite payload uses byte-packed code and scale streams:

```text
code_bytes  = ceil(weights * k / 5)
scale_bytes = ceil(ceil(weights / g) * s / 8)
payload_bpw = 8 * (code_bytes + scale_bytes) / weights
```

The ideal value is useful for comparing large tensors. The payload value is
the value to use for a concrete tensor because it includes incomplete groups
and stream padding.

## 2. The S12 family

S12 means a 12-bit log2 scale code. With k1 ternary values:

| format | code bits/value | scale density | ideal bpw |
|---|---:|---:|---:|
| `k1-g20-s12` | 1.6 | 0.6 | 2.2 |
| `k1-g15-s12` | 1.6 | 0.8 | 2.4 |
| `k1-g12-s12` | 1.6 | 1.0 | 2.6 |
| `k1-g8-s12` | 1.6 | 1.5 | 3.1 |

The first two are the original S12 choices recorded in `tk_s12.py`. A finite
stream can have a slightly higher byte payload than the ideal number because
12-bit codes are packed across pairs and incomplete groups are padded.

The same scale-density axis works at every k. For example, k5 with 10-bit
scales is approximately 8.078 ideal bpw at g128, 8.2 at g64, 8.4 at g32, and
8.6 at g20. These are format choices, not quality measurements.

## 3. Mixtures

Real models do not need one format everywhere. The designer represents a model
as a list of parts. Each part specifies:

- parameter count;
- weight k;
- weights per scale;
- scale precision;
- optional residual k, residual density, and residual precision.

For parts `i` with parameter count `n_i`, the mixture density is computed from
the total payload, not by averaging display labels:

```text
mixture_bpw = 8 * sum(payload_bytes_i) / sum(n_i)
```

This permits a k1 S12 embedding, k3 attention projections, k5 MLP blocks, and a
k8 sensitive head in one recipe without hiding the scale overhead of any part.

## 4. Autogrid integration

Autogrid remains responsible for scanning a safetensors model, measuring each
tensor against its source container's noise floor, refusing tensors that need
real placement/reconstruction, and performing reversible byte-length-guarded
patches. The Tk-Integer Designer is the format-planning layer beside that
scanner. It does not turn a density estimate into a quality claim and it does
not silently convert a model.

The app exposes the canonical k presets and the complete in-between ladder. A
user can select a preset, alter any field, add the result as one mixture part,
and repeat the process for every model region. The exported JSON is a recipe,
not a checkpoint.

## 5. Evidence boundary

The following are directly supported by the implementation:

- five-trit-per-byte code accounting;
- independent k and scale-density controls;
- S12 g20 and g15 accounting;
- finite payload accounting with byte padding;
- weighted mixture accounting;
- optional residual-plane accounting;
- Autogrid integration without changing its existing scan/convert safeguards.

The following remain open:

- model-quality impact of each scale density;
- whether a specific mixture is optimal for a model family;
- end-to-end runtime and memory under every backend;
- novelty relative to all prior grouped-scale, residual, ternary, and low-bit
  training formats.

The application should therefore report density as an engineering fact and
leave quality to the existing Autogrid measurement and later model tests.

## 6. Reproducibility

The implementation is contained in:

- `tkintegers.py` for format definitions and accounting;
- `tkintegers_ui.py` for the desktop designer;
- `autogrid.py` for the scanner/converter and integration button;
- `tests/test_tkintegers.py` for S12, padding, residual, ladder, and mixture
  checks.

No model run is implied by this paper draft. A publication version must attach
the exported recipes, exact source revision, model tensor shapes, raw Autogrid
reports, and matched quality measurements.

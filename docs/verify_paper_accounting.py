"""Check paper byte-count examples using scalar arithmetic; no model execution."""
from pathlib import Path
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tkintegers import ContainerSpec
from vendor.model_atlas.quant.formats import parse_format, storage


def global_s12_bytes(count, k, group):
    groups = (count + group - 1) // group
    return (groups * group * k + 4) // 5 + 3 * ((groups + 1) // 2)


def report():
    groups = [20, 15, 12, 8, 5, 2, 1]
    rates = [{"group": g, "nominal_bpw": 1.6 + 12 / g,
              "atlas_full_group_bpw": (8 * ((g + 4) // 5) + 12) / g} for g in groups]
    examples = []
    for rows, cols, group, expected in [(3, 23, 20, (20, 22, 33)),
                                       (1, 2000, 15, (601, 603, 603)),
                                       (1, 3, 1, (6, 7, 8))]:
        planner = ContainerSpec(1, group, 12).storage(rows * cols)["payload_bytes"]
        global_ = global_s12_bytes(rows * cols, 1, group)
        atlas = storage(rows, cols, parse_format(f"TKinteger-k1-g{group}-s12"))["total"]
        assert (planner, global_, atlas) == expected
        examples.append({"rows": rows, "columns": cols, "group": group,
                         "planner_bytes": planner, "global_bytes": global_, "atlas_bytes": atlas})
    assert storage(3, 23, parse_format("TKinteger-k1-g20-s12")) == {"payload":24,"scales":9,"total":33}
    output = {
        "kind": "analytical byte counts, not model measurements",
        "baseline": "4e3d91628302586c7a94d994641b391d656826c4",
        "s12_rates": rates, "finite_examples": examples,
        "s12_in_range_relative_bound_percent": 100 * (2 ** (24 / 4094) - 1),
        "trit_bits": 1.6, "alphabet_log2": math.log2(3),
        "packing_overhead_percent": 100 * (1.6 / math.log2(3) - 1),
        "latent_k5_r5_g64_full_group_bpw_fp16_scales": 1.6 * (5 + 5) + 32 / 64,
    }
    return output


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))

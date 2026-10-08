"""Metadata-only regressions for recipe import and checkpoint name resolution."""
import copy
import json
from pathlib import Path
import shutil
import unittest

from model_atlas import atlas_request, canonical_tensor, format_policy, review_recipe


def sample_plan():
    return {"schema_version":5,"model":"fixture","config":{"num_hidden_layers":1,"hidden_size":4,"vocab_size":8},
        "tied_embeddings":False,"precision":[
            {"path":"model.layers.0.self_attn.q_proj","kind":"weight","precision":"TKinteger-k3","group_size":4,"protection":[]},
            {"path":"model.layers.0.input_layernorm","kind":"weight","precision":"original","protection":["Automatic"]},
            {"path":"lm_head","kind":"weight","precision":"FP32","protection":[]}],
        "size_assumptions":{"scale_group_size":4,"scale_bytes_per_group":-1}}


def sample_tensors():
    return [
        {"tensor":"model.language_model.layers.0.self_attn.q_proj.weight","shape":[4,4],"dtype":"BF16"},
        {"tensor":"model.language_model.layers.0.input_layernorm.weight","shape":[4],"dtype":"BF16"},
        {"tensor":"lm_head.weight","shape":[8,4],"dtype":"BF16"},
        {"tensor":"model.visual.blocks.0.weight","shape":[4,4],"dtype":"BF16"}]


class RecipeMappingTests(unittest.TestCase):
    def test_mixed_recipe_preserves_protected_and_unlisted_tensors(self):
        plan = sample_plan()
        before = copy.deepcopy(plan)
        review = review_recipe(plan, sample_tensors())
        self.assertEqual(review["issues"], [])
        policies = review["policies"]
        self.assertEqual(policies[sample_tensors()[0]["tensor"]]["k"], 3)
        self.assertEqual(policies[sample_tensors()[1]["tensor"]]["kind"], "original")
        self.assertEqual(policies["lm_head.weight"]["dtype"], "float32")
        self.assertEqual(policies["model.visual.blocks.0.weight"]["kind"], "original")
        self.assertEqual(plan,before)

    def test_unhandled_runtime_and_factorization_are_blockers(self):
        plan = sample_plan()
        plan["precision"].append({"path":"model.layers.0.@flow.state","kind":"runtime_state","precision":"INT8"})
        plan["precision"][0]["precision"] = "NQ-binary-r64"
        review = review_recipe(plan)
        self.assertEqual(len(review["issues"]),2)

    def test_wrong_model_shapes_groups_and_missing_targets_block(self):
        plan = sample_plan()
        tensors = sample_tensors()
        tensors[0]["shape"] = [4,5]
        tensors[2]["shape"] = [9,4]
        self.assertGreaterEqual(len(review_recipe(plan,tensors)["issues"]),2)
        tensors = sample_tensors()[1:]
        self.assertTrue(any("does not match" in e for e in review_recipe(plan,tensors)["issues"]))

    def test_s12_and_scale_override_identity(self):
        plan=sample_plan()
        row={"path":"x","precision":"TKinteger-k1-g15-s12","group_size":15}
        p=format_policy(row,plan)
        self.assertEqual((p["k"],p["group"],p["scale_bits"]),(1,15,12))
        row["group_size"]=20
        with self.assertRaises(ValueError): format_policy(row,plan)
        plan["size_assumptions"]["scale_bytes_per_group"]=4
        self.assertEqual(format_policy(plan["precision"][0],plan)["scale_bits"],32)

    def test_tied_precision_conflict_is_rejected(self):
        plan=sample_plan()
        plan["tied_embeddings"]=True
        self.assertTrue(any("Tied" in e for e in review_recipe(plan)["issues"]))

    def test_vision_wrapper_is_not_text_decoder(self):
        self.assertEqual(canonical_tensor("model.visual.layers.0.weight"),"model.visual.layers.0.weight")


@unittest.skipUnless(shutil.which("node"),"Atlas engine needs Node.js")
class OriginalAtlasEngineTests(unittest.TestCase):
    def test_ten_presets_and_import_export_roundtrip(self):
        presets=atlas_request(action="presets")
        self.assertEqual(len(presets),10)
        data=atlas_request(model="Qwen/Qwen3.5-0.8B")
        plan=data["plan"]
        row=next(r for r in plan["precision"] if r["path"].endswith("0.mlp.down_proj"))
        row["precision"]="TKinteger-k1-g20-s12"
        row["group_size"]=20
        restored=atlas_request(plan=plan)
        reread=atlas_request(plan=json.loads(json.dumps(restored["plan"])))
        self.assertEqual(reread["plan"]["precision"],restored["plan"]["precision"])
        self.assertTrue(any(r["precision"]=="TKinteger-k1-g20-s12" for r in reread["plan"]["precision"]))

    def test_original_editor_state_restores_protection_and_exceptions(self):
        plan=atlas_request(model="Qwen/Qwen3.5-0.8B")["plan"]
        path="model.layers.0.linear_attn.in_proj_a"
        plan["editor_state"]={"precision_choices":{path:"TKinteger-k5"},"group_sizes":{path:32},
                              "automatic_opt_out":{path:True},"repeat_type":False}
        restored=atlas_request(plan=plan)["plan"]
        row=next(r for r in restored["precision"] if r["path"]==path)
        self.assertEqual((row["precision"],row["group_size"],row["protection"]),("TKinteger-k5",32,[]))
        other=next(r for r in restored["precision"] if r["path"]==path.replace("layers.0.","layers.1."))
        self.assertEqual(other["precision"],"original")


if __name__ == "__main__":
    unittest.main()

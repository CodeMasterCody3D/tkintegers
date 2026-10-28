"""Verify the preserved recipe without importing Torch or running a model."""
from pathlib import Path
import ast
import copy
import hashlib
import json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)}


class InjectCallbacks(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = {'packed': 'packed_linear', 'stopped': 'check_stop'}.get(node.id, node.id)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'awq_loss':
            node.args.append(ast.Name(id='packed_linear', ctx=ast.Load()))
        return node


def check_adapters(root):
    original = functions(root / 'reference/quality9_awq_qep_daptq.py')
    extracted = functions(root / 'folding.py')
    for name in ('awq_loss', 'fold_awq'):
        expected = InjectCallbacks().visit(copy.deepcopy(original[name]))
        if name == 'awq_loss':
            expected.args.args.append(ast.arg(arg='packed_linear'))
        else:
            expected.args.kwonlyargs.extend([ast.arg(arg='packed_linear'), ast.arg(arg='check_stop')])
            expected.args.kw_defaults.extend([None, None])
        require(ast.dump(expected) == ast.dump(extracted[name]),
                'AWQ source differs beyond the two dependency callbacks: ' + name)
    # Guard the actual r1 failure: the reusable writer must retain all three
    # corrected original shape conversions, not merely check final file size.
    writer = functions(root / 'storage.py')['save_matrix_like']
    def assignments(function):
        return {node.targets[0].id: ast.dump(node.value)
                for node in function.body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    old, new = assignments(original['save_matrix']), assignments(writer)
    for name in ('raw', 'scales', 'metadata'):
        require(old[name] == new[name], 'Stored-shape conversion changed: ' + name)


def verify(root):
    root = Path(root).resolve()
    lock = json.loads((root / 'provenance.json').read_text())
    for relative, expected in lock['files'].items():
        path = root / relative
        require(path.is_file(), 'Missing preserved file: ' + relative)
        require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
                'Preserved bytes changed: ' + relative)
    recipe = json.loads((root / 'recipe.json').read_text())
    contract = json.loads((root / 'reference/contract.json').read_text())
    for relative, digest in contract['new_source_lock'].items():
        require(sha(root / 'reference' / relative) == digest, 'Measured r2 source changed: ' + relative)
    codec = root.parents[1] / 'tk_codec.py'
    require(sha(codec) == contract['parent_source_lock']['tk_codec.py'] == recipe['dependencies']['tk_codec_sha256'],
            'Repository codec differs from the measured dependency')
    for name in ('format', 'group_size', 'qep_alpha', 'daptq_rounds',
                 'code_polish_steps_per_round', 'awq_grid_points', 'damping'):
        require(recipe[name] == contract[name], 'Recipe contradicts measured contract: ' + name)
    require(recipe['model']['id'] == contract['model'] and recipe['model']['revision'] == contract['revision'],
            'Model identity changed')
    calibration = root / recipe['calibration']['file']
    require(sha(calibration) == recipe['calibration']['sha256'] == contract['calibration_sha256'],
            'Calibration identity changed')
    rows = [json.loads(line) for line in calibration.read_text().splitlines()]
    require(len(rows) == recipe['calibration']['examples'] == contract['calibration_count'] == 16,
            'Calibration example count changed')
    require(len({row['id'] for row in rows}) == len(rows), 'Duplicate calibration IDs')
    require(sum(row['tokens'] for row in rows) == recipe['calibration']['tokens'] == 3102,
            'Calibration token count changed')
    for row in rows:
        require(row['tokens'] == len(row['ids']), 'Calibration token IDs lost')
        require(hashlib.sha256(row['text'].encode()).hexdigest() == row['sha256'], 'Calibration text changed')
        require(row['reasoning'] in row['text'] and row['text'].endswith('<|im_end|>\n'),
                'Complete reasoning/end token missing')
    review = json.loads((root / 'evidence/answer-review.json').read_text())
    require(review['run_id'] == contract['run_id'] == recipe['validation']['run_id'], 'Wrong result ancestry')
    require(review['weight_file_bytes_each'] == recipe['storage']['weight_file_bytes'] and review['byte_delta'] == 0,
            'Storage claim contradicts recorded result')
    require(review['question'] == recipe['validation']['question'] == 'HumanEval/127', 'Wrong question')
    bank = json.loads((root / 'evidence/final-bank-verification.json').read_text())
    bank_files = {item['name']: item['sha256'] for item in bank['files']}
    for local, remote in [('answer-review.json', 'author-review.json'), ('answers.jsonl', 'answers.jsonl'),
                          ('saved-candidate-verification.json', 'saved-candidate-verification.json'),
                          ('pilot-run-card.json', 'pilot-run-card.json'), ('full-run-card.json', 'full-run-card.json')]:
        require(sha(root / 'evidence' / local) == bank_files[remote], 'Evidence differs from independent backup: ' + local)
    check_adapters(root)
    sources = sorted(root.rglob('*.py'))
    for path in sources:
        compile(path.read_text(), str(path), 'exec')
    return {'status': 'pass', 'checked_files': len(lock['files']), 'parsed_python_files': len(sources),
            'frozen_r2_sources': len(contract['new_source_lock']),
            'awq_bodies_match_except_injected_dependencies': True,
            'corrected_storage_reshape_expressions_match': True,
            'calibration_examples': len(rows), 'calibration_tokens': 3102,
            'numerical_or_model_execution': False,
            'scope': 'Source, recipe, calibration and saved-evidence integrity only'}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))

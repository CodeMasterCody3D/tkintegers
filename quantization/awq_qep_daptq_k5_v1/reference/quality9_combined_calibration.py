"""Prepare the explicitly selected 16 complete gold training solutions remotely."""
from pathlib import Path
import hashlib
import json
import random
import shutil
import sys

ROOT = Path('/content/tk-quality9-1027')
OUT = ROOT / 'awq-qep-daptq-127-evidence'
DRIVE = Path('/content/drive/MyDrive/taardis/tk-quality9-1027') / OUT.name
REVISION = '740312add88f781978c0658806c59bc2815b9866'


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    if not Path('/content/drive/MyDrive').is_dir() or (ROOT / 'STOP').exists():
        raise RuntimeError('Assigned remote Drive required; respect human STOP')
    OUT.mkdir(exist_ok=False)
    DRIVE.mkdir(exist_ok=False)
    source = json.loads((ROOT / 'source.json').read_text())['source']
    tok = AutoTokenizer.from_pretrained(source)
    train = load_dataset('openai/gsm8k', 'main', split='train', revision=REVISION)
    test = load_dataset('openai/gsm8k', 'main', split='test', revision=REVISION)
    excluded = {r['question'].strip() for r in test}
    selected_tests = [json.loads(s)['prompt'] for s in (ROOT / 'tasks-30.jsonl').read_text().splitlines()]
    indices = list(range(len(train)))
    random.Random(20261027).shuffle(indices)
    rows = []
    for index in indices:
        row = train[index]
        question = row['question'].strip()
        if question in excluded or any(question in p for p in selected_tests):
            continue
        reasoning, sep, gold = row['answer'].rpartition('####')
        if not sep or not reasoning.strip() or not gold.strip():
            raise RuntimeError('Gold worked solution is incomplete')
        messages = [{'role': 'user', 'content': question},
                    {'role': 'assistant', 'reasoning_content': reasoning.strip(),
                     'content': '#### ' + gold.strip()}]
        text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=True,
                                       add_generation_prompt=False)
        if ('<think>\n' not in text or '</think>' not in text or
                reasoning.strip() not in text or not text.endswith('<|im_end|>\n')):
            raise RuntimeError('Actual template did not preserve the whole reasoning/answer/end token')
        ids = tok(text, add_special_tokens=False).input_ids
        rows.append({'id': f'gsm8k/train/{index}', 'question': question,
                     'reasoning': reasoning.strip(), 'gold': gold.strip(), 'text': text,
                     'ids': ids, 'tokens': len(ids), 'source_answer': row['answer'],
                     'sha256': hashlib.sha256(text.encode()).hexdigest()})
        if len(rows) == 16:
            break
    if len(rows) != 16:
        raise RuntimeError('Insufficient independent complete examples')
    path = OUT / 'calibration.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = {'dataset': 'openai/gsm8k', 'revision': REVISION, 'config': 'main',
                'split': 'train', 'seed': 20261027, 'rows': len(rows),
                'train_fingerprint': train._fingerprint, 'ids': [r['id'] for r in rows],
                'total_tokens': sum(r['tokens'] for r in rows),
                'tokens_per_example': [r['tokens'] for r in rows],
                'whole_reasoning_and_answer': True, 'all_gsm8k_test_questions_excluded': True,
                'original_30_selected_test_prompts_excluded': True,
                'thinking_source': 'Dataset gold worked solutions; no newly generated teacher traces',
                'row_quality': 'Official gold supplied; pending author reading',
                'calibration_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / 'calibration-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in OUT.iterdir():
        shutil.copy2(p, DRIVE / p.name)
    sys.path.insert(0, '/content/forge')
    from drive_sync import wait_uploaded
    if not wait_uploaded([str(DRIVE / p.name) for p in OUT.iterdir()], exact_paths=True):
        raise RuntimeError('Calibration Drive confirmation failed')
    print('CALIBRATION_READY', json.dumps(manifest), flush=True)
    print('========== DONE ==========', flush=True)


if __name__ == '__main__':
    main()

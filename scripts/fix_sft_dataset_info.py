import json
from pathlib import Path

p = Path('/root/code-assistant/LLaMA-Factory/data/dataset_info.json')
obj = json.loads(p.read_text(encoding='utf-8'))
obj['sft_pyfunc']['columns'] = {
    'prompt': 'instruction',
    'query': 'input',
    'response': 'output'
}
p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
print(obj['sft_pyfunc']['columns'])

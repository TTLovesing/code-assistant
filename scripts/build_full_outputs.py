import json
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model='/root/code-assistant/models/Qwen2.5-Coder-3B-Instruct'
sft_adapter='/root/code-assistant/training/sft_ckpt'
dpo_adapter='/root/code-assistant/training/dpo_ckpt'
subset_path=Path('/root/code-assistant/eval/results/humaneval_subset.json')
out_dir=Path('/root/code-assistant/eval/results_strict')
out_dir.mkdir(parents=True, exist_ok=True)
out_jsonl=out_dir/'qa_outputs_all_models.jsonl'
out_md=out_dir/'qa_outputs_all_models.md'

def generation_prompt(tokenizer, prompt):
    if getattr(tokenizer, 'chat_template', None):
        user_msg = (
            'Complete the given Python function.\n'
            'Rules:\n'
            '1) Output only valid Python code.\n'
            '2) Do not add tests, explanations, markdown, or extra functions.\n'
            '3) Only continue the target function body.\n\n'
            'Function skeleton:\n' + prompt
        )
        msgs=[
            {'role':'system','content':'You are a precise Python coding assistant.'},
            {'role':'user','content':user_msg},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_completion(text: str) -> str:
    text = text.replace('\r\n','\n')
    if '```' in text:
        m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL|re.IGNORECASE)
        if m:
            text = m.group(1)
    stops=[
        '\nclass ', '\nif __name__', '\nprint(', '\ndef test', '\ndef check',
        '\n# Test', '\n# test', '\n# Check', '\n**Created Question**', '\nHere',
        '\nassert ', '\n```'
    ]
    cut=len(text)
    for s in stops:
        i=text.find(s)
        if i!=-1:
            cut=min(cut,i)
    text=text[:cut].rstrip()
    m=re.search(r"(?<!\n)def\s+[A-Za-z_]\w*\s*\(", text)
    if m:
        text=text[:m.start()].rstrip()
    return text

def run_check(code: str, test_code: str, entry_point: str, timeout_sec: int = 8):
    script = f"{code}\n\n{test_code}\n\ncheck({entry_point!r})\n"
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script)
        path=f.name
    try:
        r=subprocess.run([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.remove(path)
        except OSError: pass

def generate_k(model, tok, prompt, k=3, max_new_tokens=256):
    model_in = generation_prompt(tok, prompt)
    inputs = tok(model_in, return_tensors='pt').to(model.device)
    outs=[]
    for i in range(k):
        do_sample = i>0
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.8 if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_tokens=gen[0][inputs['input_ids'].shape[1]:]
        txt=tok.decode(new_tokens, skip_special_tokens=True)
        outs.append(extract_completion(txt))
    return outs

def load_model(adapter=''):
    tok=AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    if adapter:
        model=PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok

with subset_path.open('r', encoding='utf-8') as f:
    tasks=json.load(f)

specs=[('base',''),('sft',sft_adapter),('dpo',dpo_adapter)]
all_res={}

for name,adapter in specs:
    model,tok=load_model(adapter)
    model_rows=[]
    for ex in tasks:
        prompt=ex['prompt']
        tests=ex['test']
        ep=ex['entry_point']
        cands=generate_k(model,tok,prompt,k=3,max_new_tokens=256)
        checks=[]
        for c in cands:
            full_code=prompt + c
            checks.append(run_check(full_code, tests, ep))
        model_rows.append({'task_id':ex.get('task_id',''), 'candidates':cands, 'checks':checks})
    all_res[name]=model_rows
    del model
    torch.cuda.empty_cache()

with out_jsonl.open('w', encoding='utf-8') as w:
    for i,ex in enumerate(tasks):
        row={
            'task_id': ex.get('task_id',''),
            'entry_point': ex['entry_point'],
            'prompt': ex['prompt'],
            'test': ex['test'],
            'base': all_res['base'][i],
            'sft': all_res['sft'][i],
            'dpo': all_res['dpo'][i],
        }
        w.write(json.dumps(row, ensure_ascii=False)+'\n')

with out_md.open('w', encoding='utf-8') as w:
    w.write('# Strict Eval Full Outputs (Question + Code)\n\n')
    for i,ex in enumerate(tasks, start=1):
        w.write(f"## {i}. {ex.get('task_id','')} / entry: `{ex['entry_point']}`\n\n")
        w.write('### Prompt\n\n```python\n')
        w.write(ex['prompt'])
        if not ex['prompt'].endswith('\n'):
            w.write('\n')
        w.write('```\n\n')
        for name in ('base','sft','dpo'):
            item=all_res[name][i-1]
            w.write(f"### {name.upper()}\n\n")
            for j,(code,ok) in enumerate(zip(item['candidates'], item['checks']), start=1):
                w.write(f"- cand{j} pass={ok}\n\n```python\n")
                w.write(code)
                if not code.endswith('\n'):
                    w.write('\n')
                w.write('```\n\n')
        w.write('---\n\n')

print(str(out_jsonl))
print(str(out_md))

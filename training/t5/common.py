"""Shared helpers for T5 evaluation: file I/O, the question checks and an auditing filler."""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized(text):
    return ' '.join(text.lower().split())


def checks(fields, text):
    """Joint pass = format + focus (activity kept) + novelty (retries differ) + action wording."""
    from pipeline.t5_v3 import T5V3Filler
    kind = fields['action']
    fmt = T5V3Filler.valid(text, kind, '', '')
    focus = normalized(fields['focus']) in normalized(text)
    novel = not fields['retry'] or normalized(text) not in normalized(fields['previous_question'])
    intent = True
    if kind == 'PROBE':
        intent = bool(re.search(r'enjoy|appeal|choose|want|like|satisf', text, re.I) and
                      re.search(r'experience|familiar|already know|from scratch|new to you', text, re.I))
    elif kind == 'CHALLENGE':
        intent = bool(re.search(r'which|closer|prefer|choose', text, re.I) and
                      re.search(r'statement|which of (?:these|those|the two)', text, re.I))
    return dict(format=fmt, focus=focus, novel=novel, intent=intent, joint=bool(fmt and focus and novel and intent))


def load_filler(model_path, adapter=None, device='auto', threads=4, decode=None):
    """The app's T5V3Filler, recording for every call whether T5's question or the template was used."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from pipeline.t5_v3 import T5V3Filler
    from pipeline.templates import QuestionBankFiller
    torch.set_num_threads(threads)
    resolved = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
    if resolved == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available')
    defaults = dict(max_new_tokens=64, num_beams=2, do_sample=False, no_repeat_ngram_size=3, max_input_tokens=256)
    defaults.update(decode or {})

    class Audit(T5V3Filler):
        def generate(self, text):
            started = time.perf_counter()
            tokens = self.tokenizer(text, return_tensors='pt', truncation=False)
            if tokens.input_ids.shape[1] > self.decoding['max_input_tokens']:
                raw = ''
            else:
                options = {k: v for k, v in self.decoding.items() if k != 'max_input_tokens'}
                with self.torch.inference_mode():
                    generated = self.model.generate(**tokens.to(self.device), **options)
                raw = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if self.device == 'cuda':
                self.torch.cuda.synchronize()
            self.pending.update(input=text, raw=raw, generation_seconds=time.perf_counter() - started)
            return raw

        def fill(self, action, payload):
            self.pending = dict(action=action, payload=dict(payload))
            final = super().fill(action, payload); r = self.pending; r['final'] = final
            r['source'] = 'deterministic'
            if 'raw' in r:
                fields = json.loads(r['input'].split(': ', 1)[1]); raw = r['raw']; r['fields'] = fields
                valid = self.valid(raw, fields['action'], fields['previous_question'],
                                   payload.get('task', {}).get('activity', ''))
                if fields['action'] == 'PROBE':
                    valid = valid and bool(re.search(r'experience|familiar|already know|from scratch|new to you', raw, re.I))
                r['source'] = 't5' if valid and raw in final else 'fallback'
            self.records.append(dict(r))
            return final

    f = Audit.__new__(Audit)
    QuestionBankFiller.__init__(f, read(ROOT / 'data/question_bank.json'), read(ROOT / 'data/labels.json')['type_names'])
    f.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    f.model = model.to(resolved).eval(); f.device = resolved; f.torch = torch; f.decoding = defaults; f.records = []
    return f

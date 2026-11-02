"""Real frozen generators; the reverse model never receives the original."""
from functools import partial
import json
from pathlib import Path
import sys


class DiffuSeqGenerator:
    def __init__(self, root, device="cuda", steps=2000, seeds=(425, 426, 427), checkpoint_domain="simplification"):
        import torch
        from transformers import AutoTokenizer
        self.root, self.device, self.steps, self.seeds = Path(root), device, steps, seeds
        vendor = self.root / "vendor/DiffuSeq"
        sys.path.insert(0, str(vendor))
        from basic_utils import create_model_and_diffusion
        from diffuseq.rounding import denoised_fn_round
        self.rounding = denoised_fn_round
        checkpoints = sorted((self.root / "resources/diffuseq_qqp").rglob("ema_*.pt"))
        if not checkpoints:
            raise RuntimeError("Real DiffuSeq checkpoint missing; run scripts/prepare_models.py")
        checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        self.args = json.loads((checkpoint.parent / "training_args.json").read_text())
        if checkpoint_domain == "simplification":
            checkpoint = self.root / "resources/diffuseq_ts/official_release"
            if not checkpoint.is_file():
                raise RuntimeError("Official DiffuSeq simplification checkpoint missing")
        self.args["config_name"] = str(self.root / "resources/bert_tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args["config_name"], local_files_only=True)
        self.args["vocab_size"] = len(self.tokenizer)
        # Recompute transitions for skipped timesteps. Merely setting gap>1 in
        # the old sampling loop retains adjacent-step alphas and leaves noise.
        sampling_args = {**self.args, "timestep_respacing": "" if steps == 2000 else f"ddim{steps}"}
        self.model, self.diffusion = create_model_and_diffusion(**sampling_args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.embedding = torch.nn.Embedding.from_pretrained(self.model.word_embedding.weight.detach().clone(), freeze=True).to(device)
        self.length = self.args["seq_len"]
        self.metadata = {"checkpoint": str(checkpoint.relative_to(self.root)), "sampling": "original_ancestral" if steps == 2000 else "ddim_with_upstream_spaced_diffusion",
                         "steps": steps, "seeds": list(seeds), "device": device,
                         "training_domain": checkpoint_domain,
                         "config_provenance": "Official QQP training_args; shared BERT/128-dim/2000-sqrt architecture. TS release supplies weights only; strict complete state loading verifies structural compatibility.",
                         "standardization_trained": False}

    def generate(self, text):
        import torch
        from types import SimpleNamespace
        source = self.tokenizer.encode(text, add_special_tokens=True)
        if len(source) + 1 > self.length - 16:
            raise ValueError("Clause exceeds DiffuSeq source budget; original retained without truncation")
        # Matches upstream x + [SEP] + y packaging; no reference y is provided.
        source = source + [self.tokenizer.sep_token_id]
        ids = source + [self.tokenizer.pad_token_id] * (self.length - len(source))
        tensor = torch.tensor([ids], device=self.device)
        mask = torch.tensor([[0] * len(source) + [1] * (self.length - len(source))], device=self.device)
        results = []
        for candidate_index, seed in enumerate(self.seeds, 1):
            print(f"  DiffuSeq candidate {candidate_index}/{len(self.seeds)}...", file=sys.stderr, flush=True)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                original = self.model.get_embeds(tensor)
                expanded = mask.unsqueeze(-1).expand_as(original)
                noise = torch.where(expanded == 0, original, torch.randn_like(original))
                arguments = dict(noise=noise, clip_denoised=False,
                    denoised_fn=partial(self.rounding, SimpleNamespace(**self.args), self.embedding),
                    model_kwargs={}, device=self.device, progress=False, mask=expanded, x_start=original)
                if self.steps == 2000:
                    iterator = self.diffusion.p_sample_loop_progressive(self.model, original.shape,
                        **arguments, top_p=-1., clamp_step=0, clamp_first=True)
                    for sample in iterator:
                        final = sample["sample"]
                else:
                    samples = self.diffusion.ddim_sample_loop(self.model, original.shape, **arguments, gap=1)
                    final = samples[-1]
                output = self.model.get_logits(final).argmax(dim=-1)[0, len(source):].tolist()
            if self.tokenizer.sep_token_id in output:
                output = output[:output.index(self.tokenizer.sep_token_id)]
            results.append((self.tokenizer.decode(output, skip_special_tokens=True).strip(), seed))
        return results


class Reconstructor:
    def __init__(self, path, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).eval().requires_grad_(False).to(device)
        self.device = device

    def reconstruct(self, rewrite):
        import torch
        prompt = ("Express the following statement as one natural first-person sentence. Preserve its meaning, "
                  "negation, uncertainty, preference strength and conditions. Add no information.\nStatement: " + rewrite)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.device)
        if inputs.input_ids.shape[1] > 512:
            raise ValueError("Reconstruction prompt exceeds model limit")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=96, num_beams=4, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


class FidelityEvaluator:
    def __init__(self, encoder_path, nli_path):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(2)
        self.encoder = SentenceTransformer(str(encoder_path), device="cpu", local_files_only=True)
        self.encoder.max_seq_length = 256
        self.encoder.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_path, local_files_only=True).eval().requires_grad_(False)
        labels = {str(value).casefold(): int(key) for key, value in self.nli.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError("NLI checkpoint must explicitly identify its entailment label")
        self.entailment_id = labels["entailment"]

    def evaluate(self, original, rewrite, recovered):
        import torch
        for text in [original, rewrite, recovered]:
            if len(self.encoder.tokenizer.encode(text)) > 256:
                raise ValueError("Fidelity text exceeds encoder budget; no truncated comparison allowed")
        vectors = self.encoder.encode([original, rewrite, recovered], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        pairs = [(original, rewrite), (rewrite, original), (original, recovered), (recovered, original)]
        inputs = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                                truncation=False, return_tensors="pt")
        if inputs.input_ids.shape[1] > 512:
            raise ValueError("Fidelity pair exceeds NLI budget")
        with torch.inference_mode():
            scores = self.nli(**inputs).logits.softmax(dim=-1)[:, self.entailment_id].tolist()
        direct = max(-1., min(1., float(vectors[0] @ vectors[1])))
        cycle = max(-1., min(1., float(vectors[0] @ vectors[2])))
        return direct, cycle, dict(zip(
            ["original_to_rewrite", "rewrite_to_original", "original_to_reconstruction", "reconstruction_to_original"], scores))

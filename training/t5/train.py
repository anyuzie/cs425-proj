"""Fine-tune T5-small (full or LoRA) on prompt-question pairs; keeps the checkpoint with the best validation loss.

Run from the repository root: python -m training.t5.train --model <checkpoint> --data <split dir> --output <new dir>
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def looks_local(value):
    """True for a local folder path (models/..., runs/..., absolute); False for a Hugging Face id such as google-t5/t5-small."""
    path = Path(value)
    return path.is_absolute() or path.parts[0] in ('models', 'runs', '.', '..') or '\\' in str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT/'models/t5-stage6-balanced'), help="Starting checkpoint (default: the stage-6 balanced checkpoint)")
    parser.add_argument("--data", type=Path, default=ROOT / "data/t5/standard", help="Folder with train.jsonl and validation.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output folder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--method", choices=("full", "lora"), default="full")
    parser.add_argument("--adapter", type=Path, help="Continue training the selected adapter without resetting it")
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--warmup-ratio',type=float,default=.1)
    parser.add_argument('--gradient-clip',type=float,default=1.)
    parser.add_argument('--label-smoothing',type=float,default=0.)
    parser.add_argument('--patience',type=int,default=2)
    parser.add_argument('--lora-rank',type=int,default=8)
    parser.add_argument('--lora-alpha',type=int,default=16)
    parser.add_argument('--lora-dropout',type=float,default=.05)
    parser.add_argument('--lora-targets',nargs='+',default=['q','v'])
    parser.add_argument('--backbone-dropout',type=float)
    parser.add_argument('--max-input-tokens',type=int,default=256)
    parser.add_argument('--max-target-tokens',type=int,default=64)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--deterministic',action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Epochs, batch size, threads and learning rate must be positive")
    if not 0<=args.warmup_ratio<1 or not 0<=args.label_smoothing<1 or not 0<=args.lora_dropout<1:
        parser.error('Warm-up, label smoothing and LoRA dropout must be in [0,1)')
    if args.weight_decay<0 or args.gradient_clip<=0 or args.patience<1 or args.lora_rank<1 or args.lora_alpha<1:
        parser.error('Invalid optimizer, patience or LoRA settings')
    if not all(math.isfinite(x) for x in (args.weight_decay,args.gradient_clip,args.warmup_ratio,args.label_smoothing,args.lora_dropout)):
        parser.error('All numeric settings must be finite')
    if args.backbone_dropout is not None and not 0<=args.backbone_dropout<1:parser.error('Invalid backbone dropout')
    if min(args.max_input_tokens,args.max_target_tokens)<1:parser.error('Token limits must be positive')
    if args.adapter and args.method!='lora':parser.error('An adapter may only be resumed with method=lora')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output to preserve the existing run")
    for name, value in (('--model', args.model), ('--adapter', args.adapter)):
        if value and looks_local(value) and not Path(value).exists():
            parser.error(f"{name} folder not found: {value}. Run from the repository root, and unzip the "
                         "T5 model files there so that models/t5-stage6-balanced/ exists (not inside a nested folder).")
    if not args.data.is_dir():
        parser.error(f"--data folder not found: {args.data}. Run from the repository root.")

    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is not available')
    kwargs = dict(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **kwargs).to(device)
    if args.backbone_dropout is not None:
        model.config.dropout_rate=args.backbone_dropout
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=args.backbone_dropout

    import time
    started = time.perf_counter()
    base_parameters = sum(p.numel() for p in model.parameters())
    if args.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        if args.adapter:
            config=json.loads((args.adapter/'adapter_config.json').read_text())
            if (config['r'],config['lora_alpha'],config['lora_dropout'],set(config['target_modules']))!=(args.lora_rank,args.lora_alpha,args.lora_dropout,set(args.lora_targets)):
                parser.error('Resumed adapter settings differ: start a fresh adapter for rank/alpha/dropout/target changes')
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.lora_targets, bias="none"))
        assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(dict(method=args.method, base_parameters=base_parameters, trainable_parameters=trainable), flush=True)

    def encode(rows):
        records = []
        for row in rows:
            item = tokenizer(row["input"], truncation=False)
            labels = tokenizer(text_target=row["target"], truncation=False)["input_ids"]
            if len(item["input_ids"]) > args.max_input_tokens or len(labels) > args.max_target_tokens:
                raise ValueError(f"Example {row['id']} exceeds the input/target token limit")
            item["labels"] = labels
            records.append(item)
        if not records:
            raise ValueError("Training and validation splits must not be empty")
        return records

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train = DataLoader(encode(read_rows(args.data / "train.jsonl")), batch_size=args.batch_size,
                       shuffle=True, collate_fn=collator)
    validation = DataLoader(encode(read_rows(args.data / "validation.jsonl")),
                            batch_size=args.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio)) if args.warmup_ratio else 0
    def schedule(step):
        return (step + 1) / warmup if step < warmup else max(0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    best, stale, log = float("inf"), 0, []
    print(f"Training on {device}: {len(train.dataset)} examples, {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        model.train()
        summed_loss = tokens = 0
        for step, batch in enumerate(train, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            loss = (torch.nn.functional.cross_entropy(result.logits.reshape(-1,result.logits.shape[-1]),batch['labels'].reshape(-1),ignore_index=-100,label_smoothing=args.label_smoothing)
                    if args.label_smoothing else result.loss)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; checkpoint not updated")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = (batch["labels"] != -100).sum().item()
            summed_loss += loss.item() * count
            tokens += count
            if step % 10 == 0 or step == len(train):
                print(f"epoch {epoch+1} batch {step}/{len(train)} loss={summed_loss/tokens:.4f}", flush=True)
        model.eval()
        val_loss = val_tokens = 0
        with torch.inference_mode():
            for batch in validation:
                batch = {k: v.to(device) for k, v in batch.items()}
                count = (batch["labels"] != -100).sum().item()
                val_loss += model(**batch).loss.item() * count
                val_tokens += count
        score = val_loss / val_tokens
        if not math.isfinite(score):
            raise RuntimeError("Non-finite validation loss")
        log.append(dict(epoch=epoch+1, train_loss=summed_loss/tokens, validation_loss=score,
                        optimizer_steps=(epoch+1)*len(train),learning_rate_end=optimizer.param_groups[0]['lr']))
        print(log[-1], flush=True)
        if score < best:
            best, stale = score, 0
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
        else:
            stale += 1
        manifest = dict(method=args.method, adapter_start=str(args.adapter) if args.adapter else None, trainable_parameters=trainable, base_parameters=base_parameters, elapsed_seconds=time.perf_counter()-started, model=args.model, resolved_revision=getattr(model.config, "_commit_hash", None),
                        seed=args.seed, device=device, torch=torch.__version__, transformers=transformers.__version__,
                        learning_rate=args.learning_rate, batch_size=args.batch_size, epochs_requested=args.epochs,
                        best_validation_loss=best, best_epoch=min(log, key=lambda x: x['validation_loss'])['epoch'],
                        losses=log, max_input_tokens=args.max_input_tokens, max_target_tokens=args.max_target_tokens,
                        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                        validation_definition='unsmoothed token cross-entropy, including when training uses label smoothing',
                        optimizer_defaults=optimizer.defaults,total_steps_requested=total_steps,warmup_steps=warmup,
                        effective_backbone_dropout=getattr(model.config,'dropout_rate',None),
                        continuation_semantics='weights only; optimizer, scheduler and RNG state start fresh in each invocation',
                        data_sha256={name: hashlib.sha256((args.data / f"{name}.jsonl").read_bytes()).hexdigest()
                                     for name in ("train", "validation")},
                        human_reviewed_data=False)
        (args.output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if stale >= args.patience:
            print("Early stopping: validation loss did not improve for two epochs", flush=True)
            break
    print(f"Best model saved locally to {args.output}. No files were uploaded.", flush=True)


if __name__ == "__main__":
    main()

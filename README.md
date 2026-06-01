# snc-lab

Clean, self-contained AWQ + SNC weight-only quantization. Small files, one job each.

```
snc_core.py    SNC algorithm — one pure function per paper equation (verified).
awq.py         awq_quantize_linear: AWQ scaling + grid alpha + SNC correction.
quantize.py    sequential layer-by-layer driver + CLI.
data.py        calibration loader + perplexity eval (WikiText-2 / C4).
eval_ppl.py    perplexity CLI.
selftest.py    Proposition 1 check (G>=0, |b'|<=|b|) on synthetic data.
```

`--method awq` = AWQ base; `--method snc` = AWQ + SNC (the corrected core).
SNC fixes vs the original FPRAG code: value-greedy sign (Prop. 1) and a global
top-B candidate pool across the layer instead of per-row budgets.

## Environment

```bash
conda activate llm_quant          # torch, transformers, datasets, accelerate
pip install lm_eval               # only needed for downstream tasks
python selftest.py                # sanity-check the core
```

## Quantize + eval in one go (no disk write — for small disks)

`run.py` quantizes and evaluates in the same process and saves nothing by
default. Use this on disk-constrained servers so model files never pile up.

```bash
M=Qwen/Qwen2.5-3B
python run.py --model-path $M --method snc --bits 4 --p 0.05 \
  --tasks arc_challenge,arc_easy,boolq,piqa,rte --wandb

python run.py --model-path $M --method awq  --bits 4 --tasks ... --wandb
python run.py --model-path $M --method fp16 --tasks ... --wandb    # baseline
```

Pass `--save-dir ./out/...` only if you actually want to keep the model.

## Quantize (Qwen2.5-3B, GQA, fits <=24GB)

```bash
M=Qwen/Qwen2.5-3B

# AWQ base
python quantize.py --model-path $M --method awq \
  --bits 4 --group-size 128 --n-calib 128 --calib-dataset c4 \
  --output-dir ./out/qwen3b_awq_b4

# AWQ + SNC
python quantize.py --model-path $M --method snc --p 0.05 \
  --bits 4 --group-size 128 --n-calib 128 --calib-dataset c4 \
  --output-dir ./out/qwen3b_snc_b4
```

## Perplexity + downstream tasks

`eval_ppl.py` runs WikiText-2/C4 perplexity and (optionally) the lm-eval
tasks, logging both to a single wandb run named after the model dir.

```bash
# PPL only
python eval_ppl.py --model-path ./out/qwen3b_snc_b4

# PPL + 5 lm-eval tasks, logged to wandb (run name = qwen3b_snc_b4)
python eval_ppl.py --model-path ./out/qwen3b_snc_b4 \
  --tasks arc_challenge,arc_easy,boolq,piqa,rte \
  --wandb --wandb-project snc-quant

# FP16 baseline
python eval_ppl.py --model-path $M --tasks arc_challenge,arc_easy,boolq,piqa,rte --wandb
```

Without `--wandb`, results just print to stdout (wandb/lm_eval are imported
only when actually used). For 3-bit add `--bits 3` at quantize time.

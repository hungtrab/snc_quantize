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

## Perplexity (WikiText-2 + C4)

```bash
python eval_ppl.py --model-path ./out/qwen3b_awq_b4
python eval_ppl.py --model-path ./out/qwen3b_snc_b4
python eval_ppl.py --model-path $M        # FP16 baseline
```

## Downstream — lm-eval 5 tasks

```bash
python -m lm_eval --model hf \
  --model_args pretrained=./out/qwen3b_snc_b4 \
  --tasks arc_challenge,arc_easy,boolq,piqa,rte \
  --device cuda:0 --batch_size auto \
  --output_path ./out/qwen3b_snc_b4_lmeval.json
```

For 3-bit add `--bits 3` (SNC's margin over base is larger at 3-bit). Report
the FP16 baseline alongside base/SNC for the gap-to-FP16 column.

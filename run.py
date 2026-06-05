"""Quantize + evaluate in one process, in memory — no model saved to disk.

Use this on disk-constrained servers: nothing is written unless --save-dir
is given. Logs PPL + lm-eval to a single wandb run named by config.
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize import quantize_model
from data import get_calib, eval_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--method", choices=["awq", "snc", "fp16"], default="snc")
    ap.add_argument("--bits", type=int, default=4, choices=[3, 4])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--no-snc-guard", action="store_true")
    ap.add_argument("--include-lm-head", action="store_true")
    ap.add_argument("--qk-snc", action="store_true",
                    help="use bilinear GQA alpha for q_proj/k_proj SNC")
    ap.add_argument("--attn-guard", action="store_true",
                    help="choose q/k/v/o candidates by true self-attention output loss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="snc-quant")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--save-dir", default="", help="if set, save the quantized model (off by default)")
    args = ap.parse_args()

    name = args.run_name or f"{args.method}_b{args.bits}_p{args.p}_gs{args.group_size}"
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=name)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True).eval()

    if args.method == "fp16":
        model.cuda()
    else:
        calib = get_calib(tok, args.n_calib, args.seqlen, args.seed, args.calib_dataset)
        print(f"[{args.method}] bits={args.bits} gs={args.group_size} calib={len(calib)} p={args.p}")
        quantize_model(model, calib, args.bits, args.group_size,
                       use_snc=(args.method == "snc"), p=args.p, device="cuda",
                       seed=args.seed, lam=args.lam, beta=args.beta,
                       snc_guard=not args.no_snc_guard,
                       include_lm_head=args.include_lm_head,
                       qk_snc=args.qk_snc,
                       attn_guard=args.attn_guard)
        model.cuda()

    for d in args.datasets:
        ppl = eval_ppl(model, tok, d, args.seqlen)
        print(f"{d:10s} PPL = {ppl:.4f}")
        if run: run.log({f"ppl/{d}": ppl})

    if args.tasks:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        tasks = args.tasks.split(",")
        res = simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok),
                              tasks=tasks, batch_size="auto")["results"]
        accs = []
        for t in tasks:
            acc = res[t].get("acc,none", res[t].get("acc"))
            accs.append(acc); print(f"{t:16s} acc = {acc:.4f}")
            if run: run.log({f"lm_eval/{t}": acc})
        if run: run.log({"lm_eval/avg": sum(accs) / len(accs)})

    if args.save_dir:
        model.save_pretrained(args.save_dir); tok.save_pretrained(args.save_dir)
        print(f"saved -> {args.save_dir}")
    if run: run.finish()


if __name__ == "__main__":
    main()

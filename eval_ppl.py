"""Evaluation: WikiText-2/C4 perplexity + optional lm-eval tasks.

Both go to a single wandb run per model when --wandb is set. The run name
encodes the config (defaults to the model-dir basename, e.g. qwen3b_snc_b4).
"""
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import eval_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"],
                    choices=["wikitext2", "c4"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--tasks", default="",
                    help="comma-separated lm-eval tasks, e.g. arc_challenge,arc_easy,boolq,piqa,rte")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="snc-quant")
    ap.add_argument("--run-name", default="", help="default: model-dir basename")
    args = ap.parse_args()

    run_name = args.run_name or os.path.basename(args.model_path.rstrip("/"))
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=run_name)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True).eval()

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
            accs.append(acc)
            print(f"{t:16s} acc = {acc:.4f}")
            if run: run.log({f"lm_eval/{t}": acc})
        avg = sum(accs) / len(accs)
        print(f"{'avg':16s} acc = {avg:.4f}")
        if run: run.log({"lm_eval/avg": avg})

    if run: run.finish()


if __name__ == "__main__":
    main()

"""Perplexity eval CLI: WikiText-2 and/or C4."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import eval_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--datasets", nargs="+", default=["wikitext2", "c4"],
                    choices=["wikitext2", "c4"])
    ap.add_argument("--seqlen", type=int, default=2048)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True).eval()
    for d in args.datasets:
        ppl = eval_ppl(model, tok, d, args.seqlen)
        print(f"{d:10s} PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()

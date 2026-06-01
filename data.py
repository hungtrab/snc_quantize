"""Calibration data + perplexity evaluation (WikiText-2, C4)."""
import torch
from datasets import load_dataset


def get_calib(tokenizer, n_samples=128, seqlen=2048, seed=42, dataset="c4"):
    """Return a list of (1, seqlen) token tensors, randomly sliced from docs.
    Concatenates documents until a full seqlen window is available (standard
    GPTQ/AWQ practice), so short-document corpora still yield enough samples."""
    g = torch.Generator().manual_seed(seed)
    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = (r["text"] for r in ds)
    else:  # wikitext2
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = (t for t in ds["text"] if len(t) > 0)

    out, buf = [], []
    buf_len = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids
        buf.append(ids); buf_len += ids.shape[1]
        if buf_len <= seqlen:
            continue
        cat = torch.cat(buf, dim=1)
        i = int(torch.randint(0, cat.shape[1] - seqlen, (1,), generator=g))
        out.append(cat[:, i:i + seqlen])
        buf, buf_len = [], 0
        if len(out) >= n_samples:
            break
    return out


@torch.no_grad()
def eval_ppl(model, tokenizer, dataset="wikitext2", seqlen=2048, device="cuda"):
    """Standard non-overlapping perplexity (GPTQ/AWQ convention)."""
    if dataset == "wikitext2":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    else:  # c4 validation slice
        val = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        buf, n = [], 0
        for r in val:
            buf.append(r["text"]); n += len(r["text"])
            if n > 2_000_000:
                break
        enc = tokenizer(" ".join(buf), return_tensors="pt").input_ids

    n_chunks = enc.shape[1] // seqlen
    nlls = []
    for i in range(n_chunks):
        batch = enc[:, i * seqlen:(i + 1) * seqlen].to(device)
        out = model(batch, labels=batch)
        nlls.append(out.loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n_chunks * seqlen)).item()

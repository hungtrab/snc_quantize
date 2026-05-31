"""End-to-end smoke test: tiny random GQA model, no download.
Verifies the sequential driver, hooks/kwargs, AWQ+SNC core, and a forward pass."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from quantize import quantize_model

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512,
                  num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                  max_position_embeddings=256)   # head_dim 32, GQA ratio 4
model = Qwen2ForCausalLM(cfg).eval()
calib = [torch.randint(0, 512, (1, 64)) for _ in range(4)]

q = quantize_model(model.cuda(), calib, bits=4, group_size=128, use_snc=True, p=0.1, device="cuda")
ids = torch.randint(0, 512, (1, 32)).cuda()
with torch.no_grad():
    out = q.cuda()(ids).logits
print("forward OK, logits", tuple(out.shape), "finite:", torch.isfinite(out).all().item())

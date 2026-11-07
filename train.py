import argparse, json, math, os, random
from pathlib import Path
import torch
from g1n.model import G1N
from g1n.tokenizer import ByteTokenizer

def main():
    p = argparse.ArgumentParser(description="Train G-1N from random initialization.")
    p.add_argument("--data", default="data/train.txt")
    p.add_argument("--out", default="checkpoints/g1n.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=200)
    args = p.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    if len(text.encode("utf-8")) < args.context + 2:
        raise SystemExit("Training text is too short. Add more text to " + args.data)

    tok = ByteTokenizer()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    split = max(args.context + 1, int(len(ids) * 0.9))
    train_ids = ids[:split]
    val_ids = ids[split:] if len(ids) - split > args.context else ids[:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = G1N(256, args.context, args.d_model, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batch(source):
        max_start = len(source) - args.context - 1
        starts = torch.randint(0, max_start + 1, (args.batch_size,))
        x = torch.stack([source[s:s+args.context] for s in starts]).to(device)
        y = torch.stack([source[s+1:s+args.context+1] for s in starts]).to(device)
        return x, y

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        model.train()
        xb, yb = batch(train_ids)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val_ids)
                _, val_loss = model(vx, vy)
            print(f"step {step}/{args.steps} | train {loss.item():.4f} | val {val_loss.item():.4f}")
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "context_size": args.context, "d_model": args.d_model,
                    "n_heads": args.heads, "n_layers": args.layers,
                    "vocab_size": 256
                },
                "step": step
            }, args.out)
    print("Saved checkpoint:", args.out)
    print("This is a small experimental model; quality depends on data and training.")

if __name__ == "__main__":
    main()

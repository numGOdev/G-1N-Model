import argparse, torch
from g1n.model import G1N
from g1n.tokenizer import ByteTokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/g1n.pt")
    p.add_argument("--prompt",default="Hello! ")
    p.add_argument("--tokens",type=int,default=160)
    p.add_argument("--temperature",type=float,default=0.8)
    a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    cfg=ckpt["config"]
    model=G1N(cfg["vocab_size"],cfg["context_size"],cfg["d_model"],cfg["n_heads"],cfg["n_layers"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok=ByteTokenizer()
    x=torch.tensor([tok.encode(a.prompt)],dtype=torch.long)
    out=model.generate(x,a.tokens,a.temperature)
    print(tok.decode(out[0].tolist()))

if __name__=="__main__": main()

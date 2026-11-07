from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import torch
from g1n.model import G1N
from g1n.tokenizer import ByteTokenizer

app=Flask(__name__,static_folder="web")
MODEL=None
TOKENIZER=ByteTokenizer()

def load_model():
    global MODEL
    path=Path("checkpoints/g1n.pt")
    if not path.exists(): return False
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    c=ckpt["config"]
    model=G1N(c["vocab_size"],c["context_size"],c["d_model"],c["n_heads"],c["n_layers"])
    model.load_state_dict(ckpt["model"]); model.eval()
    MODEL=model
    return True

@app.get("/")
def home(): return send_from_directory("web","index.html")

@app.get("/api/status")
def status(): return jsonify(ready=MODEL is not None, message="Model loaded" if MODEL else "Train G-1N first.")

@app.post("/api/generate")
def generate():
    if MODEL is None: return jsonify(error="No trained checkpoint found. Train the model first."),400
    body=request.get_json(silent=True) or {}
    prompt=str(body.get("prompt",""))[:2000]
    if not prompt.strip(): return jsonify(error="Enter a prompt."),400
    ids=torch.tensor([TOKENIZER.encode(prompt)],dtype=torch.long)
    with torch.no_grad(): out=MODEL.generate(ids, max_new_tokens=min(int(body.get("tokens",120)),240))
    answer=TOKENIZER.decode(out[0].tolist())
    return jsonify(text=answer[len(prompt):])

if __name__=="__main__":
    load_model()
    app.run(host="127.0.0.1",port=5000,debug=False)

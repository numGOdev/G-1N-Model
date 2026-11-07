"""Run the optional local G-1N browser demo."""
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import torch

from g1n.model import G1N
from g1n.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
MODEL = None
TOKENIZER = ByteTokenizer()


def load_model():
    global MODEL
    checkpoint_path = ROOT / "checkpoints" / "g1n.pt"
    if not checkpoint_path.is_file():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = G1N(
        config["vocab_size"],
        config["context_size"],
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    MODEL = model
    return True


@app.get("/")
def home():
    return send_from_directory(ROOT / "web", "index.html")


@app.get("/api/status")
def status():
    ready = MODEL is not None
    return jsonify(
        ready=ready,
        message="Trained model loaded." if ready else "No checkpoint found. Train G-1N first.",
    )


@app.post("/api/generate")
def generate():
    if MODEL is None:
        return jsonify(error="No trained checkpoint found. Run train.py first."), 400
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", ""))[:2000]
    try:
        token_count = int(body.get("tokens", 120))
    except (TypeError, ValueError):
        return jsonify(error="Token count must be a whole number."), 400
    if not prompt.strip():
        return jsonify(error="Please enter a prompt."), 400
    if not 1 <= token_count <= 240:
        return jsonify(error="Token count must be between 1 and 240."), 400

    input_ids = torch.tensor([TOKENIZER.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        output = MODEL.generate(input_ids, max_new_tokens=token_count)
    decoded = TOKENIZER.decode(output[0].tolist())
    return jsonify(text=decoded[len(prompt):])


if __name__ == "__main__":
    if load_model():
        print("G-1N checkpoint loaded.")
    else:
        print("No trained checkpoint found. Train G-1N before using generation.")
    app.run(host="127.0.0.1", port=5000, debug=False)

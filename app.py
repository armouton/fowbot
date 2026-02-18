"""
Flask web server for the next-word prediction tool.
"""

import os
import sys
import re
import json
import threading
import queue
import webbrowser
import zipfile
import xml.etree.ElementTree as ET
from flask import Flask, render_template, request, jsonify, Response
from model import WordPredictor


def resource_path(relative):
    """Get absolute path to a resource, works for dev and PyInstaller bundle."""
    if getattr(sys, '_MEIPASS', None):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative)


app = Flask(__name__,
            template_folder=resource_path('templates'),
            static_folder=resource_path('static'))

# Global state
predictor = WordPredictor(context_length=5, vocab_size=5000)
training_progress = queue.Queue()
is_training = False
trained_dataset_name = None

DATASETS_DIR = resource_path("datasets")


def extract_text_from_docx(filepath):
    """Extract plain text from a .docx file using only the standard library."""
    text_parts = []
    with zipfile.ZipFile(filepath, 'r') as z:
        with z.open('word/document.xml') as f:
            tree = ET.parse(f)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    for para in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        parts = []
        for run in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
            for t in run.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                if t.text:
                    parts.append(t.text)
        if parts:
            text_parts.append(''.join(parts))
    return '\n'.join(text_parts)


def sanitize_filename(name):
    """Create a safe filename from user input."""
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[\s-]+', '_', name)
    return name[:80] if name else 'uploaded'


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/datasets")
def list_datasets():
    files = sorted(f for f in os.listdir(DATASETS_DIR) if f.endswith(".txt"))
    return jsonify(files)


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    safe_name = sanitize_filename(f.filename) + ".txt"
    dest_path = os.path.join(DATASETS_DIR, safe_name)

    try:
        if ext == '.docx':
            # Save temporarily, extract text, then save as .txt
            tmp_path = dest_path + ".tmp"
            f.save(tmp_path)
            try:
                text = extract_text_from_docx(tmp_path)
            finally:
                os.remove(tmp_path)
        elif ext == '.doc':
            return jsonify({"error": "Old .doc format is not supported. Please save as .docx or .txt in Word."}), 400
        elif ext == '.rtf':
            # Basic RTF: strip RTF control codes, keep text
            raw = f.read().decode('utf-8', errors='replace')
            text = re.sub(r'[{}]', '', re.sub(r'\\[a-z]+\d*\s?', '', raw))
        else:
            # Treat everything else as plain text
            text = f.read().decode('utf-8', errors='replace')

        if len(text.split()) < 20:
            return jsonify({"error": "File has too little text (need at least 20 words)."}), 400

        with open(dest_path, 'w', encoding='utf-8') as out:
            out.write(text)

        return jsonify({"filename": safe_name, "word_count": len(text.split())})

    except zipfile.BadZipFile:
        return jsonify({"error": "Could not read .docx file. Is it a valid Word document?"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not process file: {str(e)}"}), 400


@app.route("/status")
def status():
    info = predictor.get_status()
    info["is_training"] = is_training
    info["dataset"] = trained_dataset_name
    return jsonify(info)


@app.route("/train", methods=["POST"])
def train():
    global is_training, trained_dataset_name, predictor

    if is_training:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json()
    dataset_name = data.get("dataset")
    epochs = int(data.get("epochs", 100))
    epochs = max(1, min(epochs, 500))
    learning_rate = float(data.get("learning_rate", 1.0))
    learning_rate = max(0.01, min(learning_rate, 5.0))
    hidden_size = int(data.get("hidden_size", 256))
    hidden_size = max(64, min(hidden_size, 512))
    context_length = int(data.get("context_length", 5))
    context_length = max(2, min(context_length, 20))
    architecture = data.get("architecture", "feedforward")
    if architecture not in ("feedforward", "lstm"):
        architecture = "feedforward"

    filepath = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.isfile(filepath):
        return jsonify({"error": f"Dataset not found: {dataset_name}"}), 404

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Drain any leftover messages from previous runs
    while not training_progress.empty():
        try:
            training_progress.get_nowait()
        except queue.Empty:
            break

    def on_epoch(epoch, total, loss):
        training_progress.put({"epoch": epoch, "total": total, "loss": loss})

    def run_training():
        global is_training, trained_dataset_name, predictor
        is_training = True
        try:
            predictor = WordPredictor(context_length=context_length, vocab_size=5000)
            predictor.train(text, epochs=epochs, learning_rate=learning_rate,
                            hidden_size=hidden_size, architecture=architecture,
                            on_epoch=on_epoch)
            trained_dataset_name = dataset_name
            training_progress.put({"done": True})
        except Exception as e:
            training_progress.put({"error": str(e)})
        finally:
            is_training = False

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                msg = training_progress.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("done") or msg.get("error"):
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/predict", methods=["POST"])
def predict():
    if predictor.model is None:
        return jsonify({"error": "No model trained yet. Please train a model first."}), 400

    data = request.get_json()
    context = data.get("context", "").strip()
    top_k = int(data.get("top_k", 5))
    n_words = int(data.get("n_words", 1))
    n_words = max(1, min(n_words, 500))

    if not context:
        return jsonify({"error": "Please enter some words."}), 400

    try:
        if n_words == 1:
            predictions = predictor.predict_next_words(context, top_k=top_k)
            return jsonify({
                "context": context,
                "predictions": [{"word": w, "probability": round(p, 4)} for w, p in predictions]
            })
        else:
            result = predictor.generate_sequence(context, n_words=n_words, top_k=top_k)
            return jsonify({
                "context": context,
                "generated_text": result["generated_text"],
                "generated_words": result["generated_words"],
                "steps": [
                    {
                        "predicted": s["predicted"],
                        "alternatives": [{"word": w, "probability": round(p, 4)} for w, p in s["alternatives"]]
                    }
                    for s in result["steps"]
                ]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stop", methods=["POST"])
def stop_training():
    if is_training:
        predictor.stop_training()
        return jsonify({"message": "Stop requested."})
    return jsonify({"message": "No training in progress."}), 400


if __name__ == "__main__":
    port = 5001
    url = f"http://localhost:{port}"
    print(f"\n  FowBot is running!")
    print(f"  Opening browser to: {url}\n")

    # Open browser after a short delay to let the server start
    threading.Timer(1.5, webbrowser.open, args=[url]).start()

    app.run(debug=False, port=port, threaded=True)

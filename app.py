from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the trained model (make sure model.h5 exists after training)
MODEL_PATH = "cat_dog_model.h5"

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: Model file '{MODEL_PATH}' not found. Train and save your model first.")

def predict_image(image_bytes):
    """Process image bytes and return prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)

    prediction = model.predict(img_array)[0][0]

    # class_indices from training: {'Cat': 0, 'Dog': 1}
    if prediction >= 0.5:
        label = "Dog"
        confidence = float(prediction) * 100
    else:
        label = "Cat"
        confidence = (1 - float(prediction)) * 100

    return label, confidence

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PawDetect — Cat vs Dog Classifier</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0d0d0d;
      --surface: #161616;
      --border: #2a2a2a;
      --accent-cat: #f97316;
      --accent-dog: #38bdf8;
      --text: #f0f0f0;
      --muted: #888;
      --radius: 16px;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 48px 20px;
    }

    header {
      text-align: center;
      margin-bottom: 48px;
    }

    header h1 {
      font-family: 'Syne', sans-serif;
      font-size: clamp(2.5rem, 6vw, 4rem);
      font-weight: 800;
      letter-spacing: -1px;
      background: linear-gradient(135deg, var(--accent-cat), var(--accent-dog));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    header p {
      color: var(--muted);
      font-size: 1rem;
      margin-top: 8px;
      font-weight: 300;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 36px;
      width: 100%;
      max-width: 520px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .drop-zone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 40px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      position: relative;
    }

    .drop-zone:hover, .drop-zone.drag-over {
      border-color: var(--accent-cat);
      background: rgba(249, 115, 22, 0.05);
    }

    .drop-zone input[type="file"] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }

    .drop-icon {
      font-size: 2.5rem;
    }

    .drop-zone p {
      color: var(--muted);
      font-size: 0.9rem;
      text-align: center;
    }

    .drop-zone strong {
      color: var(--text);
      font-size: 1rem;
    }

    #preview-container {
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 12px;
    }

    #preview-container img {
      width: 100%;
      max-height: 260px;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid var(--border);
    }

    #preview-container span {
      font-size: 0.8rem;
      color: var(--muted);
    }

    button#predict-btn {
      background: linear-gradient(135deg, var(--accent-cat), #fb923c);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 14px 24px;
      font-family: 'Syne', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 0.5px;
      transition: opacity 0.2s, transform 0.1s;
    }

    button#predict-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    button#predict-btn:active { transform: translateY(0); }
    button#predict-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    #result {
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      padding: 28px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #111;
      animation: fadeUp 0.4s ease;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-emoji { font-size: 3.5rem; }

    .result-label {
      font-family: 'Syne', sans-serif;
      font-size: 2rem;
      font-weight: 800;
    }

    .result-label.cat  { color: var(--accent-cat); }
    .result-label.dog  { color: var(--accent-dog); }

    .confidence-bar-wrap {
      width: 100%;
    }

    .confidence-bar-wrap p {
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 6px;
      text-align: center;
    }

    .confidence-bar-bg {
      background: var(--border);
      border-radius: 999px;
      height: 10px;
      overflow: hidden;
      width: 100%;
    }

    .confidence-bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.8s cubic-bezier(.4,0,.2,1);
    }

    .confidence-bar-fill.cat { background: var(--accent-cat); }
    .confidence-bar-fill.dog { background: var(--accent-dog); }

    .try-again-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 8px;
      padding: 8px 18px;
      font-size: 0.85rem;
      cursor: pointer;
      font-family: 'DM Sans', sans-serif;
      transition: border-color 0.2s, color 0.2s;
    }

    .try-again-btn:hover { border-color: var(--text); color: var(--text); }

    #error-msg {
      display: none;
      color: #f87171;
      font-size: 0.9rem;
      text-align: center;
      padding: 12px;
      background: rgba(248,113,113,0.08);
      border-radius: 8px;
      border: 1px solid rgba(248,113,113,0.2);
    }

    footer {
      margin-top: 40px;
      color: var(--muted);
      font-size: 0.8rem;
      text-align: center;
    }
  </style>
</head>
<body>

<header>
  <h1>PawDetect</h1>
  <p>Upload a photo — we'll tell you if it's a 🐱 or a 🐶</p>
</header>

<div class="card">
  <!-- Drop Zone -->
  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="image/*" />
    <div class="drop-icon">📁</div>
    <strong>Click or drag an image here</strong>
    <p>Supports JPG, PNG, WEBP</p>
  </div>

  <!-- Preview -->
  <div id="preview-container">
    <img id="preview-img" src="" alt="Preview" />
    <span id="file-name-label"></span>
  </div>

  <button id="predict-btn" disabled>Identify Animal</button>

  <!-- Error -->
  <div id="error-msg"></div>

  <!-- Result -->
  <div id="result">
    <div class="result-emoji" id="result-emoji"></div>
    <div class="result-label" id="result-label"></div>
    <div class="confidence-bar-wrap">
      <p id="confidence-text"></p>
      <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" id="confidence-bar"></div>
      </div>
    </div>
    <button class="try-again-btn" onclick="resetUI()">Try another image</button>
  </div>
</div>

<footer>Powered by a TensorFlow CNN trained on the PetImages dataset</footer>

<script>
  const fileInput = document.getElementById('file-input');
  const dropZone  = document.getElementById('drop-zone');
  const previewContainer = document.getElementById('preview-container');
  const previewImg = document.getElementById('preview-img');
  const fileNameLabel = document.getElementById('file-name-label');
  const predictBtn = document.getElementById('predict-btn');
  const resultDiv  = document.getElementById('result');
  const errorDiv   = document.getElementById('error-msg');

  let selectedFile = null;

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
  });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });

  function loadFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.src = ev.target.result;
      previewContainer.style.display = 'flex';
      fileNameLabel.textContent = file.name;
      predictBtn.disabled = false;
      resultDiv.style.display = 'none';
      errorDiv.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing…';
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();

      if (data.error) {
        showError(data.error);
        return;
      }

      showResult(data.label, data.confidence);

    } catch (err) {
      showError('Network error. Make sure the server is running.');
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = 'Identify Animal';
    }
  });

  function showResult(label, confidence) {
    const emoji = label === 'Cat' ? '🐱' : '🐶';
    const cls   = label.toLowerCase();

    document.getElementById('result-emoji').textContent = emoji;
    const labelEl = document.getElementById('result-label');
    labelEl.textContent = label;
    labelEl.className = `result-label ${cls}`;

    document.getElementById('confidence-text').textContent =
      `${confidence.toFixed(1)}% confidence`;

    const bar = document.getElementById('confidence-bar');
    bar.className = `confidence-bar-fill ${cls}`;
    bar.style.width = '0%';
    requestAnimationFrame(() => {
      bar.style.width = confidence.toFixed(1) + '%';
    });

    resultDiv.style.display = 'flex';
  }

  function showError(msg) {
    errorDiv.textContent = '⚠ ' + msg;
    errorDiv.style.display = 'block';
  }

  function resetUI() {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    previewImg.src = '';
    predictBtn.disabled = true;
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    predictBtn.textContent = 'Identify Animal';
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model not loaded. Please train and save your model as '{MODEL_PATH}'."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        label, confidence = predict_image(image_bytes)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    load_model()
    app.run(debug=True)
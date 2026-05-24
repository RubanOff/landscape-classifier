from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from landscape_classifier.config import load_config
from landscape_classifier.inference.predictor import Predictor
from landscape_classifier.inference.schemas import PredictionResponse

app = FastAPI(
    title="Landscape Classification API",
    version="1.0.0",
)

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor

    if _predictor is None:
        _predictor = Predictor()

    return _predictor


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Landscape Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                display: flex;
                justify-content: center;
                padding-top: 60px;
            }
            .card {
                background: white;
                width: 520px;
                padding: 32px;
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                text-align: center;
            }
            h1 {
                margin-bottom: 8px;
            }
            p {
                color: #666;
            }
            input {
                margin: 24px 0;
            }
            button {
                background: #2563eb;
                color: white;
                border: none;
                padding: 12px 22px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #1d4ed8;
            }
            img {
                max-width: 100%;
                margin-top: 20px;
                border-radius: 12px;
            }
            .result {
                margin-top: 24px;
                padding: 18px;
                background: #f0f9ff;
                border-radius: 12px;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Nature's Palette</h1>
            <p>Классификация природных ландшафтов</p>

            <input type="file" id="fileInput" accept="image/*">
            <br>
            <button onclick="predict()">Predict</button>

            <img id="preview" style="display:none;">

            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
            const fileInput = document.getElementById("fileInput");
            const preview = document.getElementById("preview");

            fileInput.addEventListener("change", () => {
                const file = fileInput.files[0];
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = "block";
                }
            });

            async function predict() {
                const file = fileInput.files[0];

                if (!file) {
                    alert("Выбери изображение");
                    return;
                }

                const formData = new FormData();
                formData.append("file", file);

                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                const result = document.getElementById("result");
                result.style.display = "block";
                result.innerHTML = `
                    <b>Class:</b> ${data.class_name}<br>
                    <b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%
                `;
            }
        </script>
    </body>
    </html>
    """


@app.get("/")
def root():
    return {"message": "Landscape Classification API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        ) from exc

    try:
        prediction: dict[str, Any] = get_predictor().predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction service is unavailable: {exc}",
        ) from exc

    return PredictionResponse(**prediction)


def run():
    import uvicorn

    cfg = load_config()

    uvicorn.run(
        "landscape_classifier.inference.main:app",
        host=cfg.inference.host,
        port=cfg.inference.port,
        reload=False,
    )

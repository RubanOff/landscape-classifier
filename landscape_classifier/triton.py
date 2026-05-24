import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from landscape_classifier.config import load_config
from landscape_classifier.export import export_production_model_to_onnx
from landscape_classifier.inference.utils import build_predict_transform
from landscape_classifier.training.logger import get_logger

logger = get_logger("landscape-triton")


def prepare_triton_repository_command() -> None:
    cfg = load_config()
    onnx_path = export_production_model_to_onnx(cfg)
    prepare_triton_repository(cfg, onnx_path)


def prepare_triton_repository(cfg, onnx_path: Path) -> Path:
    model_dir = (
        Path(cfg.triton.model_repository)
        / cfg.triton.model_name
        / str(cfg.triton.model_version)
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    triton_model_path = model_dir / "model.onnx"
    shutil.copy(
        onnx_path,
        triton_model_path,
    )

    config_path = model_dir.parent / "config.pbtxt"
    config_path.write_text(
        "\n".join(
            [
                f'name: "{cfg.triton.model_name}"',
                'backend: "onnxruntime"',
                "max_batch_size: 8",
                "input [",
                "  {",
                f'    name: "{cfg.triton.input_name}"',
                "    data_type: TYPE_FP32",
                f"    dims: [ 3, {cfg.model.image_size}, {cfg.model.image_size} ]",
                "  }",
                "]",
                "output [",
                "  {",
                f'    name: "{cfg.triton.output_name}"',
                "    data_type: TYPE_FP32",
                f"    dims: [ {cfg.model.num_classes} ]",
                "  }",
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    logger.info(f"Triton model repository prepared: {model_dir.parent}")
    return model_dir.parent


def predict_with_triton(image_path: str | Path):
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype

    cfg = load_config()
    image = Image.open(image_path).convert("RGB")
    transform = build_predict_transform(cfg.model.image_size)
    tensor = transform(image).unsqueeze(0)
    input_array = tensor.numpy().astype(np.float32)

    client = httpclient.InferenceServerClient(url=cfg.triton.url)
    infer_input = httpclient.InferInput(
        cfg.triton.input_name,
        input_array.shape,
        np_to_triton_dtype(input_array.dtype),
    )
    infer_input.set_data_from_numpy(input_array)

    infer_output = httpclient.InferRequestedOutput(cfg.triton.output_name)
    response = client.infer(
        model_name=cfg.triton.model_name,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    logits = response.as_numpy(cfg.triton.output_name)
    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=1,
    )
    confidence, prediction = torch.max(probabilities, dim=1)
    class_id = int(prediction.item())

    return {
        "class_id": class_id,
        "class_name": list(cfg.model.class_names)[class_id],
        "confidence": round(float(confidence.item()), 4),
    }


def triton_predict_command() -> None:
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: landscape-triton-predict <image_path>")

    image_path = " ".join(sys.argv[1:])
    print(predict_with_triton(image_path))

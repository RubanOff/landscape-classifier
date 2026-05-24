import os

import numpy as np
from PIL import Image

from landscape_classifier.config import load_config
from landscape_classifier.inference.utils import build_predict_transform
from landscape_classifier.training.logger import get_logger

logger = get_logger("landscape-inference")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class Predictor:
    def __init__(self, cfg=None):
        import tritonclient.http as httpclient

        self.cfg = cfg or load_config()
        self.class_names = list(self.cfg.model.class_names)
        self.transform = build_predict_transform(
            image_size=int(self.cfg.model.image_size),
        )
        self.triton_url = os.getenv("TRITON_URL", self.cfg.triton.url)
        self.model_name = self.cfg.triton.model_name
        self.input_name = self.cfg.triton.input_name
        self.output_name = self.cfg.triton.output_name
        self.client = httpclient.InferenceServerClient(url=self.triton_url)

        logger.info(
            f"Triton inference configured: url={self.triton_url}, "
            f"model={self.model_name}"
        )

    def predict(self, image: Image.Image):
        import tritonclient.http as httpclient
        from tritonclient.utils import np_to_triton_dtype

        image = image.convert("RGB")

        tensor = self.transform(image)
        input_array = tensor.unsqueeze(0).numpy().astype(np.float32)

        infer_input = httpclient.InferInput(
            self.input_name,
            input_array.shape,
            np_to_triton_dtype(input_array.dtype),
        )
        infer_input.set_data_from_numpy(input_array)
        infer_output = httpclient.InferRequestedOutput(self.output_name)

        response = self.client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output],
        )
        logits = response.as_numpy(self.output_name)
        probabilities = _softmax(logits)

        class_id = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(probabilities[0, class_id])

        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": round(confidence, 4),
        }

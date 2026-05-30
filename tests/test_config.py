from landscape_classifier.config import load_config


def test_hydra_config_composes():
    """Check that Hydra config loads."""
    cfg = load_config()

    assert cfg.model.image_size == 150
    assert cfg.training.batch_size > 0
    assert cfg.mlflow.tracking_uri == "http://127.0.0.1:8080"
    assert len(cfg.model.class_names) == 6
    assert cfg.export.onnx_path.endswith(".onnx")
    assert cfg.triton.model_name == "landscape_classifier"

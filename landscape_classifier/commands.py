def train_command() -> None:
    from landscape_classifier.training.train import train

    train()


def export_onnx_command() -> None:
    from landscape_classifier.export import export_onnx

    export_onnx()


def prepare_triton_command() -> None:
    from landscape_classifier.triton import prepare_triton_repository_command

    prepare_triton_repository_command()


def triton_predict_command() -> None:
    from landscape_classifier.triton import triton_predict_command as run_predict

    run_predict()


def api_command() -> None:
    from landscape_classifier.inference.main import run as run_api

    run_api()

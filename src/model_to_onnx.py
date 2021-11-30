import torch

from utils.helpers import create_config_parser

from src.registry import Registry


def main(
        model_path,
        model_class,
        model_params,
        output_path
):
    input_shape = (model_params['input_width'], model_params['input_height'])
    dummy_input = torch.randn(
        1, 3, input_shape[0], input_shape[1], requires_grad=True)
    model_checkpoint = torch.load(model_path)

    model = Registry.MODELS[model_class](
        input_shape=input_shape, **model_params
    )

    model.load_state_dict(model_checkpoint)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        output_path + ".onnx",
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == "__main__":
    config = create_config_parser()
    main(**config)

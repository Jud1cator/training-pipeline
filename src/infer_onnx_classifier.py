import os
import numpy as np
import onnx
import onnxruntime

from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms

from src.utils.transforms import ResizePad
from metrics.confusion_matrix import ConfusionMatrix


def load_image_for_inference(img_path, input_shape):
    pil_img = Image.open(img_path)
    transform = transforms.Compose([
        ResizePad(input_shape),
        transforms.ToTensor(),
    ])
    return transform(pil_img).unsqueeze(0)


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(params):
    classes = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    input_shape = (32, 32)
    onnx_model = onnx.load(params.model)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(params.model)

    cm = ConfusionMatrix(len(classes))

    for c in classes:
        for img_name in os.listdir(os.path.join(params.images, c)):
            input_tensor = load_image_for_inference(
                os.path.join(params.images, c, img_name), input_shape
            )
            ort_inputs = {
                ort_session.get_inputs()[0].name: to_numpy(input_tensor)
            }
            ort_outs = ort_session.run(None, ort_inputs)[0]
            pred = np.argmax(softmax(ort_outs))
            cm.update(pred, int(c[1])-1)
    print(cm.get_confusion_matrix())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--images")
    parser.add_argument("--model")
    params = parser.parse_args()
    main(params)

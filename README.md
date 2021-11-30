# Training Pipeline for Computer Vision Neural Networks

Training pipeline is a tool that helps automate the experiments with training neural
networks for computer vision tasks, such us Image Classification and Object Detection.
It implements the standard ways of training neural networks for Image Classification 
and Object Detection on custom datasets. The training procedure is configured via `.yml`
files, which provides a transparent overview of all parameters.

The code of training procedures is not universal - there are always some tweaks that
you may want to try while training your models. You can easily extend the functionality
thanks to the modular structure of the source code, which enables to add new models, losses, metrics,
data parsers and even the whole tasks (Semantic Segmentation and Instance Segmentation are coming).

## Repository structure:

This repository is organized in a following folder structure:

- `configs` - a folder for storing training procedure configurations. Contains example configs which
with possible fields and values.

- `runs` - information about runs, tensorboard logs, model checkpoints and evaluation results will 
be stored here after a training procedure is completed.

- `src` - all source code files

The source code is organized in a following folder structure:

- `data_modules` - module which contains subclasses of the `LightningDataModule` class. Used to
perform all data related operations.

- `losses` - TBD module for custom loss functions.

- `metrics` - module which contains `AbstractMetric` class and its subclasses. These classes are meant 
as containers and aggregators of different metrics that may be collected during training procedure.

- `models` - module which contains `AbstractModelWrapper` class (a subclass of `torch.nn.Module`).
Any Pytorch neural network which is subclass of `Module` or `AbstractModelWrapper` can be added here
to be used in a training procedure.

- `tasks` - module which contains subclasses of `LightningModule` which wrap up any model from `models`
module for corresponded task, defining its training procedure.

- `utils` - all helpful unclassified code goes here

All launching scripts (like `train.py`) go to the root of `src`.

## How to install

- Clone the repository
- Create and activate the virtual environment. This is important, because you
don't want to mess with packages versions which may be incompatibe with ones you
already have in you system.
Here is how you can do it using `venv` module in Python 3:

    `python3 -m venv /path/to/new/virtual/environment`

- Install requirements:

    `pip install -r requirements.txt`

__WARNING__: you may need to install different versions of `torch` and `torchvision`
packages depending on you CUDA version. For that, refer to the specific version
which are compatible with your CUDA version here: https://download.pytorch.org/whl/torch_stable.html

You need to __MANUALLY__ install needed version of `torch` and `torchvision`, for example
for CUDA 11.1:

    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111

## How to run

To run training pipeline, put your `yaml` config to `configs` folder and provide it to `src/train.py` script:

`python3 src/train.py -c configs/mnist_classification_with_perceptron.yml`

## Config structure

To simplify the process of training neural networks and searching the optimal hyperparameters you
can tweak all parts of training procedure in a single config file once all additionally needed features 
are implemented. It helps with tracking parameter values used for training and tuning them by 
collecting them all in one place. Below is how a sample config for running a training of image classifier
can look like:
```
run_params:
  name: example_classification
  seed: 1

datamodule:
  name: ClassificationDataModule
  params:
    data_dir: "/path/to/train/dataset"
    test_data_dir: "/path/to/test/dataset/if/provided"
    input_shape: [100, 100]
    val_split: 0.1
    test_split: 0.1
    batch_size: 32
    use_weighted_sampler: False
    pin_memory: True

train_transforms:
  - name: HorizontalFlip
    params:
      p: 0.5
  - name: ToTensor

val_transforms:
  - name: ToTensor


task:
  name: ClassificationTask
  params:
    visualize_first_batch: True
    network:
      name: EfficientNetLite0
      params:
        pretrained: True
    loss:
      name: CrossEntropyLoss
      params:
        is_weighted: False
    metrics:
      - name: Precision
      - name: Recall
    optimizer:
      name: Adam
      params:
        lr: 0.001

callbacks:
  - name: ModelCheckpoint
    params:
      monitor: val_precision
      mode: 'max'
      verbose: True

trainer_params:
  max_epochs: 100
  gpus: 1

export_params:
  output_name: example_detection
  to_onnx: False
```

It outlines all parameters of the training procedure: data parameters, 
transformations, model and optimizer hyperparameters, loss and metrics to collect.
Callbacks can be set to monitor the procedure, such as checkpoint monitor or early stopping.
Moreover, you can train your model on multiple GPUs by simply setting the trainer's `gpus`
parameter to the number of GPUs (Thanks to wonderful Pytorch Lightning). Finally, the trained model
can be automatically converted to ONNX format to facilitate its future deployment. ONNX can be
easily converted to such frameworks as TensorRT or OpenVINO for fast inference on GPU and CPU.

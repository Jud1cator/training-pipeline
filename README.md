# Training Pipeline for Computer Vision Neural Networks

## Repository structure:

- `configs` - all .yml configs which are used to launch the system

- `runs` - information about runs, tensorboard logs, model checkpoints and evaluation results

- `src` - all source code files

## Source code structure

Module `src` contains all project submodules and high level scripts.

Submodules:

- `data_modules` - subclasses of pytorch_lightning.LigtningDataModule which manage datasets for specific task (12.07.2021: only classification_datamodule is implemented)

- `losses` - definitions of loss functions (12.07.2021: nothing is implemented).
- `metrics` - definitions of evaluation metric for different tasks (12.07.2021: only confusion_matrix is implemented).
- `models` - subclasses of torch.nn.Module with definitions neural network architectures (12.07.2021: bit_vehicle_classifier_net and simple_net are implemented).
- `tasks` - subclasses of pytorch_lightning.LigtningModule which define the training and evaluation process for specific task (12.07.2021: only classification_task is implemented).
- `utils` - helper scripts.

Scripts:

- `infer_onnx_classifier.py` - script for classifier inference converted to ONNX format.
- `model_to_onnx.py` - script to convert model weights in Pytorch format (JSON) to ONNX.
- `test.py` - script to run evaluation pipeline.
- `train.py` - main script, runs training and evaluation pipelines.
- `visualize_inference.py` - script to visualize inference of detection network.

Note: Helper scripts were written for early versions of training pipeline and might need refactoring before use.

## How to run

To run training pipeline, you need to put .yml config to `configs` folder and provide it to `src/train.py` script.

Example: `python3 src/train.py -c config_for_classification_task.yml`

## Config structure

To simplify the process of training neural networks, one can do it simply interacting with config once all needed code is implemented. It helps with tracking parameter values used for training and tuning them by collecting them all in one place. Below is the example of config for trainig NN for image classification:
```
run_params:
  name: effnet_lite0_classes=6_resolution=96
  seed: 1

datamodule:
  name: ClassificationDataModule
  params:
    data_dir: "/home/dragon/Projects/edge_vision/dvc/sbc_core/Datasets/vehicle_classification_6/data"
    test_data_dir: "/home/dragon/Projects/edge_vision/dvc/sbc_core/Datasets/dooh_classification_3"
    classes:
      - c1
      - c2
      - c3
      - c4
      - c5
      - c6
    input_shape: [96, 96]
    val_split: 0.2
    batch_size: 32
    use_weighted_sampler: True
    pin_memory: True
    num_workers: 4

task:
  name: ClassificationTask
  params:
    debug: True

    network:
      name: EfficientNetLite0
      params:
        pretrained: True

    loss:
      name: CrossEntropyLoss
      weighted: False
      params:
        {}

    optimizer:
      name: Adam
      params:
        lr: 0.001

    scheduler:
      name: ReduceLROnPlateau
      params:
        patience: 4
        verbose: True

callbacks:
  EarlyStopping:
    monitor: val_loss
    mode: min
    patience: 10
    min_delta: 0.01
    verbose: True
  ModelCheckpoint:
    {}

trainer_params:
  max_epochs: 100
  gpus: 1

export_params:
  output_name: "effnet_lite0_classes=6_resolution=96"
  batch_size: 32
```

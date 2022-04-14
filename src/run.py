import logging
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.registry import Registry
from src.utils.config_validation import Config
from src.utils.helpers import create_config_parser


def prepare_run(name, seed):
    pl.seed_everything(seed)
    all_runs_dir = Path('./runs')
    date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    run_name = '_'.join([str(date), f'_{name}'])
    run_dir = all_runs_dir / run_name
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    test_res_dir = run_dir / 'results'
    test_res_dir.mkdir(exist_ok=True, parents=True)
    weights_dir = run_dir / 'trained_models'
    weights_dir.mkdir(exist_ok=True, parents=True)
    return run_dir, checkpoints_dir, test_res_dir, weights_dir


def main(
        run_params: dict,
        datamodule: dict,
        train_transforms: list,
        val_transforms: list,
        task: dict,
        callbacks: list,
        trainer_params: dict,
        export_params: dict
):
    # Initializing run folders
    run_dir, ckpt_dir, res_dir, weights_dir = prepare_run(**run_params)

    # Scanning src module to fill the registry
    Registry.init_modules()

    # Initializing transforms
    train_tf_list = []
    for tf in train_transforms:
        transform = Config(**tf)
        train_tf_list.append(Registry.TRANSFORMS[transform.name](**transform.params))
    val_tf_list = []
    for tf in val_transforms:
        transform = Config(**tf)
        val_tf_list.append(Registry.TRANSFORMS[transform.name](**transform.params))

    # Initializing datamodule
    datamodule_config = Config(**datamodule)
    dm = Registry.DATA_MODULES[datamodule_config.name](
        **datamodule_config.params,
        train_transforms=train_tf_list,
        val_transforms=val_tf_list
    )

    # Initializing task
    task_config = Config(**task)
    task = Registry.TASKS[task_config.name](datamodule=dm, res_dir=res_dir, **task_config.params)

    # Initializing callbacks
    callbacks_list = []
    checkpoint_callback = None
    for callback in callbacks:
        callback = Config(**callback)
        if callback.name == 'ModelCheckpoint':
            callback.params['dirpath'] = ckpt_dir
        callbacks_list.append(Registry.CALLBACKS[callback.name](**callback.params))
        if callback.name == 'ModelCheckpoint':
            checkpoint_callback = callbacks_list[-1]

    # Initializing logger
    logger = TensorBoardLogger(save_dir=str(run_dir), name='tb_logs')

    # Initialing trainer
    trainer = Trainer(
        **trainer_params,
        logger=logger,
        callbacks=callbacks_list
    )

    # --- TRAINING ---
    if dm.train_set is not None:
        trainer.fit(task, datamodule=dm)

        # Restoring the best tracked model from checkpoint
        if checkpoint_callback is not None:
            model_state_dict = torch.load(checkpoint_callback.best_model_path)['state_dict']
            state_dict_keys = list(model_state_dict.keys())
            for k in state_dict_keys:
                model_state_dict['.'.join(k.split('.')[1:])] = model_state_dict.pop(k)
            task.model.load_state_dict(model_state_dict)

        # Saving only weights from the best model
        weights_fname = '.'.join([export_params['output_name'], 'pt'])
        torch.save(task.model.state_dict(), weights_dir / weights_fname)

    # --- EVALUATION ---
    task.eval()
    if dm.test_set is not None:
        trainer.test(task, datamodule=dm)
    else:
        logging.log(1, 'No test data were provided, skipping testing')

    # --- EXPORT ---
    if export_params.get('to_onnx', False):
        batch_size = export_params.get('batch_size', 1)
        opset_version = export_params.get('opset_version', 9)
        onnx_fname = export_params['output_name'] + '.onnx'
        dummy_input = dm.create_dummy_input(batch_size)
        torch.onnx.export(
            task.model,
            dummy_input,
            weights_dir / onnx_fname,
            input_names=['input'],
            output_names=['output'],
            opset_version=opset_version
        )


if __name__ == '__main__':
    _, config = create_config_parser()
    main(**config)

from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils.config_validation import Config
from utils.helpers import create_config_parser
from registry import Registry


def prepare_run(name, seed):
    torch.manual_seed(seed)
    all_runs_dir = Path("./runs")
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
    run_dir, ckpt_dir, res_dir, weights_dir = prepare_run(**run_params)

    Registry.init_modules()

    datamodule = Config(**datamodule)

    train_tf_list = []
    for tf in train_transforms:
        transform = Config(**tf)
        train_tf_list.append(Registry.TRANSFORMS[transform.name](**transform.params))
    val_tf_list = []
    for tf in val_transforms:
        transform = Config(**tf)
        val_tf_list.append(Registry.TRANSFORMS[transform.name](**transform.params))

    task = Config(**task)

    dm = Registry.DATA_MODULES[datamodule.name](
        **datamodule.params,
        train_transforms=train_tf_list,
        val_transforms=val_tf_list
    )

    task = Registry.TASKS[task.name](datamodule=dm, res_dir=res_dir, **task.params)

    callbacks_list = []
    for callback in callbacks:
        callback = Config(**callback)
        if callback.name == 'ModelCheckpoint':
            callback.params['dirpath'] = ckpt_dir
        callbacks_list.append(Registry.CALLBACKS[callback.name](**callback.params))

    logger = TensorBoardLogger(save_dir=str(run_dir), name='tb_logs')

    trainer = Trainer(
        **trainer_params,
        logger=logger,
        callbacks=callbacks_list
    )

    trainer.fit(task, datamodule=dm)

    weights_fname = '.'.join([export_params['output_name'], 'json'])
    torch.save(task.model.state_dict(), weights_dir / weights_fname)

    task.eval()
    if dm.test_set:
        trainer.test(task, datamodule=dm)

    if export_params['to_onnx']:
        input_shape = tuple(datamodule.params['input_shape'])
        batch_size = export_params.get('batch_size', 1)

        dummy_input = torch.randn(batch_size, 3, input_shape[0], input_shape[1])

        onnx_fname = export_params['output_name'] + ".onnx"
        torch.onnx.export(
            task.model,
            dummy_input,
            weights_dir / onnx_fname,
            input_names=['input'],
            output_names=['output'],
        )


if __name__ == '__main__':
    _, config = create_config_parser()
    main(**config)

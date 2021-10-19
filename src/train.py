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
    return run_dir, checkpoints_dir, test_res_dir


def main(
        run_params: dict,
        datamodule: dict,
        task: dict,
        callbacks: list,
        trainer_params: dict,
        export_params: dict
):
    run_dir, ckpt_dir, res_dir = prepare_run(**run_params)

    Registry.init_modules()

    datamodule = Config(**datamodule)
    task = Config(**task)

    export_path = Path("./trained_models")
    export_path.mkdir(exist_ok=True)

    dm = Registry.DATA_MODULES[datamodule.name](**datamodule.params)

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
    torch.save(task.net.state_dict(), export_path / weights_fname)

    task.eval()
    trainer.test(task, datamodule=dm)

    input_shape = tuple(datamodule.params['input_shape'])

    dummy_input = torch.randn(export_params['batch_size'], 3, input_shape[0], input_shape[1])

    onnx_fname = export_params['output_name'] + ".onnx"
    torch.onnx.export(
        task.net,
        dummy_input,
        export_path / onnx_fname,
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    _, config = create_config_parser()
    main(**config)

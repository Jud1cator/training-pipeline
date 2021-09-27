from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from registry import Registry
from utils.helpers import create_config_parser


def prepare_run(name, seed):
    torch.manual_seed(seed)
    all_runs_dir = Path("./runs")
    date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    run_name = str(date) + f'_{name}'
    run_dir = all_runs_dir / run_name
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    test_res_dir = run_dir / 'results'
    test_res_dir.mkdir(exist_ok=True, parents=True)
    return run_dir, checkpoints_dir, test_res_dir


def main(
        run_params,
        datamodule,
        task,
        callbacks,
        trainer_params,
        export_params
):
    run_dir, ckpt_dir, res_dir = prepare_run(**run_params)

    Registry.init_modules()

    logger = TensorBoardLogger(save_dir=str(run_dir), name='tb_logs')

    input_shape = tuple(datamodule['params']['input_shape'])

    export_path = Path("./trained_models")
    export_path.mkdir(exist_ok=True)

    dm = Registry.DATA_MODULES[datamodule['name']](**datamodule['params'])

    task = Registry.TASKS[task['name']](
        datamodule=dm, res_dir=res_dir, **task['params']
    )

    callbacks_list = []
    for name, params in callbacks.items():
        if name == 'ModelCheckpoint':
            params['dirpath'] = ckpt_dir
        callbacks_list.append(Registry.CALLBACKS[name](**params))

    trainer = Trainer(
        **trainer_params,
        logger=logger,
        callbacks=callbacks_list
    )

    trainer.fit(task, datamodule=dm)

    weights_fname = export_params['output_name'] + ".json"
    torch.save(task.net.state_dict(), export_path / weights_fname)

    task.eval()

    if dm.has_test_data:
        trainer.test(task, datamodule=dm)

    dummy_input = torch.randn(
        export_params['batch_size'], 3, input_shape[0], input_shape[1])

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

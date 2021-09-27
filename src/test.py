from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer

from registry import Registry
from utils.helpers import create_config_parser
from utils.visualization import plot_confusion_matrix


def prepare_run(name, seed):
    torch.manual_seed(seed)
    all_runs_dir = Path("./runs")
    date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    run_name = str(date) + f'_{name}'
    run_dir = all_runs_dir / run_name
    test_res_dir = run_dir / 'results'
    test_res_dir.mkdir(exist_ok=True, parents=True)
    return run_dir, test_res_dir


def main(
        run_params,
        datamodule,
        task,
        trainer_params
):
    run_dir, res_dir = prepare_run(**run_params)

    Registry.init_modules()

    dm = Registry.DATA_MODULES[datamodule['name']](**datamodule['params'])

    task = Registry.TASKS[task['name']](datamodule=dm, res_dir=res_dir, **task['params'])

    trainer = Trainer(**trainer_params)

    trainer.test(task, datamodule=dm)


if __name__ == '__main__':
    _, config = create_config_parser()
    main(**config)

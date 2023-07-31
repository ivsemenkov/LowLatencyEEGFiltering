import logging
import os
import eco2ai
import hydra
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from eeg_nn_filtering.builders import build_experiment
from general_utilities.constants import PROJECT_ROOT
from general_utilities.utils import get_devices, make_deterministic, save_json


@hydra.main(version_base=None, config_path='',
            config_name='')
def run_full_experiment(cfg: DictConfig):
    """
    Train one forecasting model on one dataset

    Parameters
    ----------
    cfg : DictConfig
                Configuration file for an experiment
    """
    dataset_path = os.path.join(PROJECT_ROOT, cfg['dataset_relative_path'])
    output_root = os.getcwd()
    experiment_name = cfg['model']
    start_from_checkpoint = cfg['start_from_checkpoint']
    tensorboard_path = os.path.join(output_root, 'TrainLogs')
    preferred_device = cfg['device']

    device, trainer_accelerator, trainer_devices = get_devices(preferred_device)
    with open_dict(cfg['trainer_params']):
        cfg['trainer_params']['accelerator'] = trainer_accelerator
        cfg['trainer_params']['devices'] = trainer_devices

    system, train_loader, val_loader, test_loader = build_experiment(config=cfg,
                                                                     dataset_path=dataset_path,
                                                                     split_ds_into_chunks=cfg['split_ds_into_chunks'])

    if start_from_checkpoint is not None:
        assert isinstance(start_from_checkpoint, str)
        ckpt_path = os.path.join(PROJECT_ROOT, start_from_checkpoint)
    else:
        ckpt_path = None

    checkpoint_params = cfg['checkpoint_params']
    fname = checkpoint_params['filename']

    checkpoint_callback = ModelCheckpoint(dirpath=tensorboard_path,
                                          **checkpoint_params)
    checkpoint_callback.CHECKPOINT_NAME_LAST = f'{fname}-last'
    logger = TensorBoardLogger(tensorboard_path,
                               name=experiment_name)

    callbacks = [checkpoint_callback]

    early_stopping_params = cfg['early_stopping_params']

    if early_stopping_params is not None:

        early_stopping_callback = EarlyStopping(**early_stopping_params)

        callbacks.append(early_stopping_callback)

    trainer_params = cfg['trainer_params']

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=tensorboard_path,
        **trainer_params
    )

    tracker = eco2ai.Tracker(
        project_name=experiment_name,
        experiment_description=f'training of {experiment_name}',
        file_name='train_emission.csv',
        alpha_2_code=cfg['alpha_2_country_code']
    )

    tracker.start()

    trainer.fit(system,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)

    tracker.stop()

    if test_loader is not None:

        best_model_path = trainer.checkpoint_callback.best_model_path

        checkpoint = torch.load(best_model_path)
        system.load_state_dict(checkpoint['state_dict'])
        system.eval()

        tracker = eco2ai.Tracker(
            project_name=experiment_name,
            experiment_description=f'testing {experiment_name}',
            file_name='test_emission.csv',
            alpha_2_code=cfg['alpha_2_country_code']
        )

        tracker.start()

        test_metrics = trainer.test(system,
                                    dataloaders=test_loader)[0]

        tracker.stop()

        test_metrics['best_model_path'] = best_model_path

        save_json(test_metrics, os.path.join(output_root, 'test_metrics.json'))


if __name__ == '__main__':
    logging.getLogger('apscheduler.executors.default').propagate = False
    logging.getLogger('apscheduler.scheduler').propagate = False
    make_deterministic(seed=0)
    run_full_experiment()

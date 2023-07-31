import argparse
import copy
import logging
import os
from pathlib import Path
import eco2ai
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from eeg_nn_filtering.builders import build_experiment
from general_utilities.constants import PROJECT_ROOT, RESULTS_ROOT
from general_utilities.utils import get_devices, make_deterministic, read_json, save_json


def add_data_parameters(dict_cfg: dict, dataset_type: str, person_idx: int, batch_size: int, num_workers: int):
    """
    Add specific parameters to the dataset parameters for a person

    Parameters
    ----------
    dict_cfg : dict
                Dictionary with configuration
    dataset_type : str
                Type of dataset. Expected here 'multiperson_real_data'
    person_idx : int
                Id of a person
    batch_size : int
                Batch size
    num_workers : int
                Amount of DataLoader workers

    Returns
    -------
    person_dict : dict
                Configuration for a person
    """
    person_dict = copy.deepcopy(dict_cfg)
    person_dict['dataset_type'] = dataset_type
    person_dict['trainer_params']['max_epochs'] = 200  # Amount of fine-tune epochs
    person_dict['dataset_common_params'] = {
        'input_size': 5024,
        'output_size': 4000,
        'gap': 3250,
        'ids_to_use': [person_idx],
        'shuffle_dataset': False,
        'maximal_per_person_size': 90
    }

    person_dict['train_dataset_specific_params'] = {'first_observation_idx': 0,
                                                    'amount_of_observations': 45}

    person_dict['val_dataset_specific_params'] = {'first_observation_idx': 45,
                                                  'amount_of_observations': 9}

    person_dict['test_dataset_specific_params'] = {'first_observation_idx': 54,
                                                   'amount_of_observations': 36}

    person_dict['dataloader_params'] = {
        'batch_size': batch_size,
        'drop_last': False,
        'num_workers': num_workers
    }

    person_dict['system_params']['save_predictions'] = True

    return person_dict


def test_from_checkpoint(root_path: str, epoch_name: str, dataset_path: str, batch_size: int, num_workers: int,
                         preferred_device: str):
    """
    Test a trained model on a real EEG dataset

    Parameters
    ----------
    root_path : str
                Path to the experiment which has trained model from project root.
                Example: results/outputs/TCNModel/YYYY-MM-DD/00-00-00/
    dataset_path : str
                Path to the multi-person EEG dataset
    batch_size : int
                Batch size for testing
    num_workers : int
                Amount of loader workers for prediction
    preferred_device : str
                Preferred device as 'cpu' or 'gpu'. If 'gpu' chosen, but cuda is not available for PyTorch will use
                'cpu'
    """
    # Parameters of the real_data_250Hz.npz dataset
    dataset_name = 'multiperson_real_data'
    dataset_type = 'multiperson_real_data'
    people_indices = [0, 11, 22, 30, 49]
    sampling_rate = 250

    basename = os.path.basename(root_path)
    if basename == '':
        root_path, _ = os.path.split(root_path)
    work_dir = os.path.join(PROJECT_ROOT, root_path)
    cfg_path = os.path.join(work_dir, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(cfg_path)
    if epoch_name == 'best':
        jsonfile = read_json(os.path.join(work_dir, 'test_metrics.json'))
        ckpt_path = jsonfile['best_model_path']
    else:
        ckpt_path = os.path.join(work_dir, 'TrainLogs', f'{epoch_name}.ckpt')
    experiment_name = cfg['model']
    split_ds_into_chunks = cfg['split_ds_into_chunks']
    dict_cfg = OmegaConf.to_container(cfg)
    # Making sure, normalization will happen
    dict_cfg['norm_by_train_std'] = True
    device, trainer_accelerator, trainer_devices = get_devices(preferred_device)
    dict_cfg['trainer_params']['accelerator'] = trainer_accelerator
    dict_cfg['trainer_params']['devices'] = trainer_devices

    output_root, timestamp = os.path.split(root_path)
    output_root, datestamp = os.path.split(output_root)
    _, model_name = os.path.split(output_root)
    output_dir = os.path.join(model_name, datestamp, timestamp)
    output_name = 'test_metrics'
    metrics_root = os.path.join(RESULTS_ROOT, 'test_results', output_dir)
    Path(metrics_root).mkdir(parents=True, exist_ok=True)
    eco2ai_dir = os.path.join(metrics_root, 'eco2ai_reports')
    Path(eco2ai_dir).mkdir(parents=True, exist_ok=True)

    output_extension = '.json'
    counter = 0
    output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extension
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extension

    full_test_metrics = {}

    for person_idx in people_indices:

        person_dict = add_data_parameters(dict_cfg=dict_cfg,
                                          dataset_type=dataset_type,
                                          person_idx=person_idx,
                                          batch_size=batch_size,
                                          num_workers=num_workers)

        system, train_loader, val_loader, test_loader = build_experiment(config=OmegaConf.create(person_dict),
                                                                         dataset_path=dataset_path,
                                                                         split_ds_into_chunks=split_ds_into_chunks)

        checkpoint = torch.load(ckpt_path)
        system.load_state_dict(checkpoint['state_dict'])

        trainer_params = person_dict['trainer_params']

        early_stopping_params = person_dict['early_stopping_params']
        my_stopper = EarlyStopping(**early_stopping_params)

        subject_work_dir = os.path.join(work_dir, model_name, f'finetune_subj_{person_idx}')

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(subject_work_dir, 'checkpoints'),
            save_last=True,
            monitor="val_loss",
            filename="best-{epoch}-{val_loss:.2f}")
        checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}"

        trainer = Trainer(
            callbacks=[checkpoint_callback, my_stopper],
            default_root_dir=os.path.join(subject_work_dir, 'logs'),
            **trainer_params
        )

        trainer.fit(system,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=None)

        best_model_path = trainer.checkpoint_callback.best_model_path

        checkpoint = torch.load(best_model_path)
        system.load_state_dict(checkpoint['state_dict'])
        system.eval()

        system.reset_test_metrics()
        system.reset_predictions()

        tracker = eco2ai.Tracker(
            project_name=experiment_name,
            experiment_description=f'full test of {experiment_name}',
            file_name=os.path.join(eco2ai_dir, f'full_test_emission_subj_{person_idx}') + '_' + str(counter) + '.csv',
            alpha_2_code=cfg['alpha_2_country_code']
        )

        tracker.start()

        test_metrics = trainer.test(system,
                                    dataloaders=test_loader)[0]

        tracker.stop()

        full_test_metrics[f'subj_{person_idx}'] = system.test_metrics
        full_test_metrics[f'subj_{person_idx}_ckpt'] = best_model_path

        npz_root = os.path.join(RESULTS_ROOT, 'saved_predictions', output_dir)
        Path(npz_root).mkdir(parents=True, exist_ok=True)
        npz_path = os.path.join(npz_root, output_name) + f'_person_{person_idx}_{counter}.npz'
        system.save_predictions_npz(npz_path)

    full_test_metrics['start_ckpt_path'] = ckpt_path
    full_test_metrics['dataset_type'] = dataset_type
    full_test_metrics['dataset_name'] = dataset_name
    full_test_metrics['sampling_rate'] = sampling_rate

    save_json(full_test_metrics, output_path)


if __name__ == '__main__':
    logging.getLogger('apscheduler.executors.default').propagate = False
    logging.getLogger('apscheduler.scheduler').propagate = False
    make_deterministic(seed=0)
    parser = argparse.ArgumentParser(description='Testing one filtering model on a real multi-person dataset.')
    parser.add_argument('-p', '--path', type=str, help='Relative path to the experiment which has trained model '
                                                       'from project root. '
                                                       'Example: '
                                                       'results/outputs/TCNFilteringModel/YYYY-MM-DD/00-00-00/',
                        required=True)
    parser.add_argument('-r', '--dataset_path', type=str,
                        help='Relative path to the "real_data_250Hz.npz" dataset from project root.'
                             'Example: Datasets/real_data_250Hz.npz', required=True)
    parser.add_argument('-e', '--epoch_name', type=str,
                        help='Name of the epoch to use. If epoch_name==best will use the epoch marked as '
                             'best_model_path in the PROJECT_ROOT/path/test_metrics.json', required=False,
                        default='best')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Batch size to use.', required=False,
                        default=512)
    parser.add_argument('-w', '--num_workers', type=int,
                        help='Number of workers to use in DataLoader.', required=False,
                        default=0)
    parser.add_argument('-d', '--preferred_device', type=str,
                        help='Name of the preferred device to use. If "gpu" is chosen, but cuda is not available '
                             'it will ignore this argument and use "cpu"', choices=['cpu', 'gpu'], required=False,
                        default='gpu')
    args = parser.parse_args()
    test_from_checkpoint(root_path=args.path, epoch_name=args.epoch_name, dataset_path=args.dataset_path,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         preferred_device=args.preferred_device)

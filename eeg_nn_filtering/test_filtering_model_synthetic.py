import argparse
import logging
import math
import os
import typing as tp
from pathlib import Path
import eco2ai
import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from eeg_nn_filtering.builders import build_experiment
from general_utilities.constants import PROJECT_ROOT, RESULTS_ROOT
from general_utilities.utils import get_devices, make_deterministic, read_json, save_json


def single_test(root_path: str, epoch_name: str, dataset_name: str, frequency: tp.Optional[float], duration_test: float,
                fs: float, amount_of_observations: int, filter_band: tp.Optional[tuple[float, float]],
                noise_coeff: float, batch_size: int, num_workers: int, preferred_device: str):
    """
    Test a model on a single specified dataset

    Parameters
    ----------
    root_path : str
                Path to the experiment which has trained model from project root.
                Example: results/outputs/TCNModel/YYYY-MM-DD/00-00-00/
    epoch_name : str
                Name of the epoch for testing. If epoch_name == 'best' will take the best validation epoch from json
    dataset_name : str
                Synthetic dataset name to generate. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                'state_space_white', 'state_space_pink']
    frequency : float
                Central frequency for dataset generation. Use None for filtered_pink as it relies on filter_band
    duration_test : float
                Amount of seconds per single time-series in a test dataset
    fs : float
                Sampling rate for the data
    amount_of_observations : int
                Amount of time-series (observations) in a resultant dataset. Total dataset size is
                duration * fs * amount_of_observations samples. Each time-series will be generated independently
    filter_band : tuple (band_low, band_high)
                Band to filter ground truth in for the filtered_pink dataset. Use None for others
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    batch_size : int
                Batch size for testing
    num_workers : int
                Amount of DataLoader workers
    preferred_device : str
                Preferred device as 'cpu' or 'gpu'. If 'gpu' chosen, but cuda is not available for PyTorch will use
                'cpu'
    """
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
    input_chunk_length = cfg['model_params']['input_chunk_length']
    dataset_type = 'generated'

    device, trainer_accelerator, trainer_devices = get_devices(preferred_device)

    with open_dict(cfg):
        cfg['dataset_type'] = dataset_type
        cfg['dataset_common_params'] = {
            'input_size': int(duration_test * fs + input_chunk_length),
            'output_size': int(duration_test * fs),
            'duration': duration_test + 1 + math.ceil(input_chunk_length / fs),
            'fs': fs,
            'dataset_name': dataset_name,
            'frequency': frequency,
            'filter_band': filter_band,
            'amount_of_observations': amount_of_observations,
            'noise_coeff': noise_coeff,
            'rng_seed': 0
        }

        for prefix in ['train', 'val', 'test']:
            cfg[f'{prefix}_dataset_specific_params'] = {}

        cfg['dataloader_params'] = {
            'batch_size': batch_size,
            'drop_last': False,
            'num_workers': num_workers
        }

        cfg['system_params']['save_predictions'] = True
        cfg['trainer_params']['accelerator'] = trainer_accelerator
        cfg['trainer_params']['devices'] = trainer_devices

    system, train_loader, val_loader, test_loader = build_experiment(config=cfg,
                                                                     dataset_path=None,
                                                                     split_ds_into_chunks=False)

    checkpoint = torch.load(ckpt_path)
    system.load_state_dict(checkpoint['state_dict'])
    system.eval()

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

    trainer_params = cfg['trainer_params']

    trainer = Trainer(
        logger=False,
        **trainer_params
    )

    system.reset_test_metrics()
    system.reset_predictions()

    tracker = eco2ai.Tracker(
        project_name=experiment_name,
        experiment_description=f'full test of {experiment_name}',
        file_name=os.path.join(eco2ai_dir, 'full_test_emission') + '_' + str(counter) + '.csv',
        alpha_2_code=cfg['alpha_2_country_code']
    )

    tracker.start()

    test_metrics = trainer.test(system,
                                dataloaders=test_loader)[0]

    tracker.stop()

    full_test_metrics = system.test_metrics

    npz_root = os.path.join(RESULTS_ROOT, 'saved_predictions', output_dir)
    Path(npz_root).mkdir(parents=True, exist_ok=True)
    npz_path = os.path.join(npz_root, output_name) + '_' + str(counter) + '.npz'
    system.save_predictions_npz(npz_path)

    full_test_metrics['ckpt_path'] = ckpt_path
    full_test_metrics['dataset_type'] = dataset_type
    full_test_metrics['dataset_name'] = dataset_name
    full_test_metrics['sampling_rate'] = fs
    full_test_metrics['noise_coeff'] = noise_coeff
    if dataset_name in ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink']:
        full_test_metrics['frequency'] = frequency
    elif dataset_name == 'filtered_pink':
        full_test_metrics['filter_band'] = filter_band

    save_json(full_test_metrics, output_path)


def test_from_checkpoint(root_path: str, dataset_name: str, epoch_name: str, batch_size: int, num_workers: int,
                         preferred_device: str):
    """
    Test a model on a single specified dataset

    Parameters
    ----------
    root_path : str
                Path to the experiment which has trained model from project root.
                Example: results/outputs/TCNModel/YYYY-MM-DD/00-00-00/
    dataset_name : str
                Synthetic dataset name to generate. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                'state_space_white', 'state_space_pink']
    epoch_name : str
                Name of the epoch for testing. If epoch_name == 'best' will take the best validation epoch from json
    batch_size : int
                Batch size for testing
    num_workers : int
                Amount of DataLoader workers
    preferred_device : str
                Preferred device as 'cpu' or 'gpu'. If 'gpu' chosen, but cuda is not available for PyTorch will use
                'cpu'
    """
    duration_test = 20  # size in seconds of one time series
    amount_of_observations = 180  # amount of time series in a generated dataset
    fs = 250  # sampling rate

    if dataset_name in ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink']:
        # These datasets are generated using the central frequency. Therefore, it must be given.
        # Filter band is not used as there is no filtering here.
        frequency = 10
        filter_band = None
    elif dataset_name == 'filtered_pink':
        # Filtered pink ground truth is made by generating a sample of pink noise and filtering it with FIR filter in
        # the filter band. Therefore, it is needed here. Central frequency is not used here.
        filter_band = (8, 12)
        frequency = None
    else:
        raise ValueError

    # Noise level (NL) from the paper equals to noise_coeff for a white noise.
    # Noise level (NL) from the paper equals to 10 * noise_coeff for a pink noise.
    if dataset_name in ['filtered_pink', 'state_space_pink']:
        # There are multiple tests of the same checkpoint for multiple noise levels to check the robustness to the
        # pink noise.
        noise_coeffs = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
    else:
        noise_coeffs = [1.0]

    for noise_coeff in noise_coeffs:

        single_test(root_path=root_path, epoch_name=epoch_name, dataset_name=dataset_name,
                    frequency=frequency, duration_test=duration_test, fs=fs,
                    amount_of_observations=amount_of_observations,
                    filter_band=filter_band, noise_coeff=noise_coeff,
                    batch_size=batch_size, num_workers=num_workers,
                    preferred_device=preferred_device)


if __name__ == '__main__':
    logging.getLogger('apscheduler.executors.default').propagate = False
    logging.getLogger('apscheduler.scheduler').propagate = False
    make_deterministic(seed=0)
    parser = argparse.ArgumentParser(description='Testing one filtering model on a synthetic generated dataset.')
    parser.add_argument('-p', '--path', type=str, help='Path to the experiment which has trained model '
                                                       'from project root. '
                                                       'Example: '
                                                       'results/outputs/TCNFilteringModel/YYYY-MM-DD/00-00-00/',
                        required=True)
    parser.add_argument('-n', '--dataset_name', type=str,
                        choices=['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink', 'filtered_pink'],
                        help='Name of the required dataset for generation', required=True)
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
    test_from_checkpoint(root_path=args.path, dataset_name=args.dataset_name, epoch_name=args.epoch_name,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         preferred_device=args.preferred_device)

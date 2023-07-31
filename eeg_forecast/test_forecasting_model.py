import argparse
import logging
import os
import typing as tp
from pathlib import Path
import eco2ai
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf, open_dict
from eeg_forecast.available_models import AVAILABLE_DARTS_MODELS
from eeg_forecast.compatible_dataset import DartsCompatibleGeneratedDataset
from eeg_forecast.darts_test_metrics import get_test_metrics, forecast_with_darts, save_predictions_gt
from general_utilities.constants import PROJECT_ROOT, RESULTS_ROOT
from general_utilities.utils import get_devices, make_deterministic, save_json


def single_test(root_path: str, best_test_epoch: bool, dataset_name: str, frequency: tp.Optional[float],
                duration_test: float, fs: float, amount_of_observations: int,
                filter_band: tp.Optional[tuple[float, float]], noise_coeff: float, batch_size: int,
                preferred_device: str, num_loader_workers: int):
    """
    Test a model on a single specified dataset

    Parameters
    ----------
    root_path : str
                Path to the experiment which has trained model from project root.
                Example: results/outputs/TCNModel/YYYY-MM-DD/00-00-00/
    best_test_epoch : bool
                If True uses best validation epoch. If False uses last epoch
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
    preferred_device : str
                Preferred device as 'cpu' or 'gpu'. If 'gpu' chosen, but cuda is not available for PyTorch will use
                'cpu'
    num_loader_workers : int
                Amount of loader workers for prediction
    """
    basename = os.path.basename(root_path)
    if basename == '':
        root_path, _ = os.path.split(root_path)
    work_dir = os.path.join(PROJECT_ROOT, root_path)
    cfg_path = os.path.join(work_dir, '.hydra', 'config.yaml')
    cfg = OmegaConf.load(cfg_path)
    gt_lag = cfg['dataset_common_params']['gt_lag']
    model_params = cfg['model_params']
    model_name = cfg['model']
    opt_params = cfg['opt_params']
    sch_params = cfg['sch_params']
    additional_params = cfg['additional_params']
    loss_func = nn.MSELoss()
    pl_trainer_kwargs = OmegaConf.to_container(cfg['pl_trainer_kwargs'])

    assert gt_lag >= model_params['output_chunk_length'], 'Else, covariates size should be different'

    _, trainer_accelerator, trainer_devices = get_devices(preferred_device)
    if pl_trainer_kwargs['accelerator'] != trainer_accelerator:
        with open_dict(pl_trainer_kwargs):
            pl_trainer_kwargs['accelerator'] = trainer_accelerator
            pl_trainer_kwargs['devices'] = trainer_devices

    model = AVAILABLE_DARTS_MODELS[model_name](optimizer_cls=optim.AdamW,
                                               optimizer_kwargs=opt_params,
                                               lr_scheduler_cls=optim.lr_scheduler.ReduceLROnPlateau,
                                               lr_scheduler_kwargs=sch_params,
                                               model_name=model_name,
                                               pl_trainer_kwargs=pl_trainer_kwargs,
                                               loss_fn=loss_func,
                                               batch_size=batch_size,
                                               **additional_params,
                                               **model_params)

    test_ds = DartsCompatibleGeneratedDataset(input_chuck_length=model_params['input_chunk_length'],
                                              duration=duration_test,
                                              fs=fs,
                                              dataset_name=dataset_name,
                                              frequency=frequency,
                                              filter_band=filter_band,
                                              amount_of_observations=amount_of_observations,
                                              gt_lag=gt_lag,
                                              noise_coeff=noise_coeff,
                                              rng_seed=0)

    best_model = model.load_from_checkpoint(model_name=model_name, work_dir=work_dir, best=best_test_epoch)
    ckpt_path = best_model.load_ckpt_path

    output_root, timestamp = os.path.split(root_path)
    output_root, datestamp = os.path.split(output_root)
    _, model_name = os.path.split(output_root)
    output_dir = os.path.join(model_name, datestamp, timestamp)
    output_name = 'test_metrics'
    metrics_root = os.path.join(RESULTS_ROOT, 'test_results', output_dir)
    Path(metrics_root).mkdir(parents=True, exist_ok=True)
    eco2ai_dir = os.path.join(metrics_root, 'eco2ai_reports')
    Path(eco2ai_dir).mkdir(parents=True, exist_ok=True)

    output_extention = '.json'
    counter = 0
    output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extention
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extention

    tracker = eco2ai.Tracker(
        project_name=model_name,
        experiment_description=f'full test of {model_name}',
        file_name=os.path.join(eco2ai_dir, 'full_test_emission') + '_' + str(counter) + '.csv',
        alpha_2_code=cfg['alpha_2_country_code']
    )

    tracker.start()

    prediction = forecast_with_darts(best_model, test_ds, gt_lag, num_loader_workers, batch_size)

    tracker.stop()

    test_target_ts, test_covariates = test_ds.get_target_and_cov()

    detailed_errors_dict, errors_dict = get_test_metrics(gt_ts_list=test_target_ts,
                                                         predict_ts_list=prediction,
                                                         amount_of_steps=gt_lag,
                                                         input_chunk_length=model_params['input_chunk_length'])

    full_test_metrics = {'errors_per_batch_reduced_dict': errors_dict,
                         'errors_detailed_dict': detailed_errors_dict}

    print(errors_dict)

    npz_root = os.path.join(RESULTS_ROOT, 'saved_predictions', output_dir)
    Path(npz_root).mkdir(parents=True, exist_ok=True)
    npz_path = os.path.join(npz_root, output_name) + '_' + str(counter) + '.npz'
    save_predictions_gt(filename=npz_path, gt_ts_list=test_target_ts, predict_ts_list=prediction,
                        amount_of_steps=gt_lag, input_chunk_length=model_params['input_chunk_length'])

    full_test_metrics['ckpt_path'] = ckpt_path
    full_test_metrics['dataset_name'] = dataset_name
    full_test_metrics['dataset_type'] = 'generated'
    full_test_metrics['gt_lag'] = gt_lag
    full_test_metrics['sampling_rate'] = fs
    full_test_metrics['noise_coeff'] = noise_coeff
    if dataset_name in ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink']:
        full_test_metrics['frequency'] = frequency
    elif dataset_name == 'filtered_pink':
        full_test_metrics['filter_band'] = filter_band

    save_json(full_test_metrics, output_path)


def test_from_checkpoint(root_path: str, dataset_name: str, batch_size: int, num_loader_workers: int,
                         preferred_device: str):
    """
    Test a trained model on a required dataset

    Parameters
    ----------
    root_path : str
                Path to the experiment which has trained model from project root.
                Example: results/outputs/TCNModel/YYYY-MM-DD/00-00-00/
    dataset_name : str
                Synthetic dataset name to generate. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                'state_space_white', 'state_space_pink']
    batch_size : int
                Batch size for testing
    num_loader_workers : int
                Amount of loader workers for prediction
    preferred_device : str
                Preferred device as 'cpu' or 'gpu'. If 'gpu' chosen, but cuda is not available for PyTorch will use
                'cpu'
    """

    duration_test = 20  # size in seconds of one time series
    amount_of_observations = 180  # amount of time series in a generated dataset
    fs = 250  # sampling rate
    best_test_epoch = True  # when True will use the best validation epoch, when False will use last available epoch

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

        single_test(root_path=root_path, best_test_epoch=best_test_epoch, dataset_name=dataset_name,
                    frequency=frequency, duration_test=duration_test, fs=fs,
                    amount_of_observations=amount_of_observations,
                    filter_band=filter_band, noise_coeff=noise_coeff,
                    batch_size=batch_size, preferred_device=preferred_device,
                    num_loader_workers=num_loader_workers)


if __name__ == '__main__':
    logging.getLogger('apscheduler.executors.default').propagate = False
    logging.getLogger('apscheduler.scheduler').propagate = False
    make_deterministic(seed=0)
    parser = argparse.ArgumentParser(description='Testing one forecasting model on a synthetic generated dataset.')
    parser.add_argument('-p', '--path', type=str, help='Path to the experiment which has trained model '
                                                       'from project root. '
                                                       'Example: results/outputs/TCNModel/YYYY-MM-DD/00-00-00/',
                        required=True)
    parser.add_argument('-n', '--dataset_name', type=str,
                        choices=['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink', 'filtered_pink'],
                        help='Name of the required dataset for generation', required=True)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Batch size to use.', required=False,
                        default=2048)
    parser.add_argument('-w', '--num_workers', type=int,
                        help='Number of workers to use in DataLoader.', required=False,
                        default=0)
    parser.add_argument('-d', '--preferred_device', type=str,
                        help='Name of the preferred device to use. If "gpu" is chosen, but cuda is not available '
                             'it will ignore this argument and use "cpu"', choices=['cpu', 'gpu'], required=False,
                        default='gpu')
    args = parser.parse_args()
    test_from_checkpoint(root_path=args.path, dataset_name=args.dataset_name, batch_size=args.batch_size,
                         num_loader_workers=args.num_workers, preferred_device=args.preferred_device)

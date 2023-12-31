import datetime
import os
import typing as tp
from pathlib import Path
import numpy as np
import torch
from general_utilities.constants import RESULTS_ROOT
from general_utilities.metrics import calculate_batched_circstd, calculate_batched_correlation, calculate_batched_plv
from general_utilities.utils import save_json


def calculate_metrics(complex_pred: np.ndarray, complex_gt: np.ndarray):
    """
    Calculate all metrics for the specified complex predicted analytic signal and complex ground truth analytic signal

    Parameters
    ----------
    complex_pred : ndarray
                Complex predicted filtered analytic signal
    complex_gt : ndarray
                Complex ground truth analytic signal

    Returns
    -------
    detailed_errors_dict : dict
                Dictionary which has lists of error for appropriate metrics
                ['correlation', 'plv', 'circstd', 'circstd_degrees']
    aggregated : dict
                Dictionary which has aggregated error values per appropriate metric
                ['correlation', 'plv', 'circstd', 'circstd_degrees']
    predictions : dict
                Dictionary with predictions for the dataset (observations filtered with a classic filter). It has
                keys 'complex_pred', 'complex_gt' for aligned filtration results and ground truth respectively. It also
                has keys 'envelope_pred' and 'envelope_gt' for their respective envelopes
    """

    batch_size = complex_pred.shape[0]
    assert complex_gt.shape[0] == batch_size

    complex_pred = torch.tensor(complex_pred)
    complex_gt = torch.tensor(complex_gt)

    assert torch.is_complex(complex_pred)
    assert torch.is_complex(complex_gt)

    envelope_pred = torch.abs(complex_pred)
    envelope_gt = torch.abs(complex_gt)

    correlations_per_batch = calculate_batched_correlation(envelope_pred, envelope_gt)
    plvs_per_batch = calculate_batched_plv(complex_pred, complex_gt)
    circstd_per_batch = calculate_batched_circstd(complex_pred, complex_gt)
    circstd_per_batch_degrees = circstd_per_batch * 180. / np.pi

    detailed_errors_dict = {
        'correlation': correlations_per_batch.cpu().numpy().tolist(),
        'plv': plvs_per_batch.cpu().numpy().tolist(),
        'circstd': circstd_per_batch.cpu().numpy().tolist(),
        'circstd_degrees': circstd_per_batch_degrees.cpu().numpy().tolist()
    }

    aggregated = {}

    for metric_name, errors in detailed_errors_dict.items():
        assert len(errors) == batch_size
        stats = {
            'min': np.min(errors),
            'max': np.max(errors),
            'mean': np.mean(errors),
            'median': np.median(errors)
        }
        aggregated[metric_name] = stats

    predictions = {'complex_pred': complex_pred.cpu().numpy(),
                   'complex_gt': complex_gt.cpu().numpy(),
                   'envelope_pred': envelope_pred.cpu().numpy(),
                   'envelope_gt': envelope_gt.cpu().numpy()}

    return detailed_errors_dict, aggregated, predictions


def save_results(detailed_errors_dict: dict, predictions: dict, dataset_type: str, dataset_name: str,
                 frequency: tp.Optional[float], filter_band: tp.Optional[tuple[float, float]], sampling_rate: float,
                 noise_coeff: float, model_name: str, model_params: dict):
    """
    Save all results of classic filters. Metrics and additional information are saved in results/test_tesults
    in json file, while predictions themselves are saved in results/saved_predictions in npz file

    Parameters
    ----------
    detailed_errors_dict : dict
                Dictionary which has lists of error for appropriate metrics
                ['correlation', 'plv', 'circstd', 'circstd_degrees']
    predictions : dict
                Dictionary with predictions for the dataset (observations filtered with a classic filter). It has
                keys 'complex_pred', 'complex_gt' for aligned filtration results and ground truth respectively. It also
                has keys 'envelope_pred' and 'envelope_gt' for their respective envelopes
    dataset_type : str
                Type of the dataset. Will be only used for writing to json with results as a hint
    dataset_name : str
                Name of the dataset. One of ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink',
                'filtered_pink']. Will be only used for writing to json with results as a hint
    frequency : float
                Central frequency for the dataset. Will be only used for writing to json with results as a hint.
                Suggested to use None if dataset is filtered_pink (relies on filter_band, not central frequency during
                generation) or multiperson_real_data (ground truth central frequency is unknown)
    filter_band : tuple (freq_low, freq_high)
                Use only for filtered_pink dataset. This dataset is generated by adding a sample of pink noise to
                another sample of pink noise filtered in the filter_band. For others use None.
                Will be only used for writing to json with results as a hint
    sampling_rate : float
                Sampling rate of the data. Will be only used for writing to json with results as a hint
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise Level.
                Use None for multiperson_real_data as here the noise comes from the data rather that being added during
                generation. Will be only used for writing to json with results as a hint
    model_name : str
                Name of the model to write to appropriate model folder in the results directory. Your path to saved
                results will be results/test_results/model_name/YYYY-MM-DD/00-00-00
    model_params : dict
                A dictionary with model parameters which lead to respective predictions and metrics
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_date, current_time = now.split()
    current_time = current_time.replace(':', '-')

    full_test_metrics = {'detailed_errors_dict': detailed_errors_dict}

    output_dir = os.path.join(model_name, current_date, current_time)
    output_name = 'test_metrics'
    metrics_root = os.path.join(RESULTS_ROOT, 'test_results', output_dir)
    Path(metrics_root).mkdir(parents=True, exist_ok=True)

    output_extention = '.json'
    counter = 0
    output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extention
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extention

    npz_root = os.path.join(RESULTS_ROOT, 'saved_predictions', output_dir)
    Path(npz_root).mkdir(parents=True, exist_ok=True)
    npz_path = os.path.join(npz_root, output_name) + '_' + str(counter) + '.npz'
    np.savez(npz_path, **predictions)

    full_test_metrics['dataset_type'] = dataset_type
    full_test_metrics['dataset_name'] = dataset_name
    full_test_metrics['sampling_rate'] = sampling_rate
    full_test_metrics['noise_coeff'] = noise_coeff
    if dataset_name in ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink']:
        full_test_metrics['frequency'] = frequency
    elif dataset_name == 'filtered_pink':
        full_test_metrics['filter_band'] = filter_band

    for param_name, param_value in model_params.items():
        full_test_metrics[param_name] = param_value

    save_json(full_test_metrics, output_path)


def save_multipers_results(per_person_dict: dict, dataset_type: str, dataset_name: str, sampling_rate: float,
                           model_name: str):
    """
    Save all results of classic filters for multi-person real EEG experiments.
    Metrics and additional information are saved in results/test_tesults in json file,
    while predictions themselves are saved in results/saved_predictions in npz file.
    It saves predictions while remembering which metrics and predictions relate to which person id.

    Parameters
    ----------
    per_person_dict : dict
                A dictionary with keys which represent different person id. Under each such key there is a
                subdictionary 'detailed_errors_dict' which has lists of error for appropriate metrics
                ['correlation', 'plv', 'circstd', 'circstd_degrees'] and 'predictions' with predictions for the dataset
                (observations filtered with a classic filter)
    dataset_type : str
                Type of the dataset. In this case expected 'multiperson_real_data'.
                Will be only used for writing to json with results as a hint
    dataset_name : str
                Name of the dataset. In this case expected 'multiperson_real_data'.
                Will be only used for writing to json with results as a hint
    sampling_rate : float
                Sampling rate of the data. Will be only used for writing to json with results as a hint
    model_name : str
                Name of the model to write to appropriate model folder in the results directory. Your path to saved
                results will be results/test_results/model_name/YYYY-MM-DD/00-00-00
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_date, current_time = now.split()
    current_time = current_time.replace(':', '-')

    output_dir = os.path.join(model_name, current_date, current_time)
    output_name = 'test_metrics'
    metrics_root = os.path.join(RESULTS_ROOT, 'test_results', output_dir)
    Path(metrics_root).mkdir(parents=True, exist_ok=True)

    output_extension = '.json'
    counter = 0
    output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extension
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(metrics_root, output_name) + '_' + str(counter) + output_extension

    full_test_metrics = {}

    for person_idx, person_data in per_person_dict.items():
        detailed_errors_dict = person_data['detailed_errors_dict']
        predictions = person_data['predictions']
        full_test_metrics[f'subj_{person_idx}'] = detailed_errors_dict

        npz_root = os.path.join(RESULTS_ROOT, 'saved_predictions', output_dir)
        Path(npz_root).mkdir(parents=True, exist_ok=True)
        npz_path = os.path.join(npz_root, output_name + f'_person_{person_idx}') + '_' + str(counter) + '.npz'
        np.savez(npz_path, **predictions)

    full_test_metrics['dataset_type'] = dataset_type
    full_test_metrics['dataset_name'] = dataset_name
    full_test_metrics['sampling_rate'] = sampling_rate

    save_json(full_test_metrics, output_path)

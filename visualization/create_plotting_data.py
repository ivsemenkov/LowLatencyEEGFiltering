import argparse
import os
import typing as tp
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from tqdm import tqdm
from general_utilities.constants import RESULTS_ROOT
from general_utilities.metrics import calculate_batched_circstd, calculate_batched_correlation, calculate_batched_plv
from general_utilities.utils import read_json
from visualization.visualization_utils import get_algorithms_list, get_full_metric_name, get_model_id, get_model_name, \
    get_real_data_npz_filepaths, extract_multiperson_predictions_gt, extract_predictions_gt, test_dataset_names


def create_parents(filepath: str):
    """
    Creates all parent directories for the file if they do not yet exist

    Parameters
    ----------
    filepath : str
                Path to the file for which parent directories have to be created
    """
    root = os.path.dirname(filepath)
    Path(root).mkdir(parents=True, exist_ok=True)


def crosscorr(c1: np.ndarray, c2: np.ndarray, sr: float, shift_nsamp: int = 500):
    """
    Calculate normed cross-correlation between two sequences

    Parameters
    ----------
    c1 : ndarray
                First array of data
    c2 : ndarray
                Second array of data
    sr : float
                Sampling rate of both arrays
    shift_nsamp : int
                Maximal shift in samples for cross-correlation

    Returns
    -------
    times : ndarray
                Values of checked shifts in milliseconds
    corr_values : ndarray
                Normed cross-correlation values
    """
    assert len(c1) == len(c2), "Arrays must be of the same length"
    c1, c2 = c1.copy(), c2[shift_nsamp:-shift_nsamp].copy()
    c2 -= c2.mean()
    c1s = np.array([c1[i: i + len(c2)] for i in range(len(c1) - len(c2) + 1)])
    c1s = c1s - np.mean(c1s, axis=1, keepdims=True)
    cc1 = np.array([np.sum(c1s[i] ** 2) for i in range(len(c1) - len(c2) + 1)])
    cc2 = np.dot(c2, c2)
    times = np.arange(-shift_nsamp, shift_nsamp + 1) / sr
    dot_products = np.dot(c1s, c2)
    return times, (dot_products / np.sqrt(cc1 * cc2))[::-1]


def calculate_batched_crosscorr(complex_pred: np.ndarray, complex_gt: np.ndarray, lag_shift: int, sampling_rate: float,
                                dataset_name: str):
    """
    Calculate maximal cross-correlation value with respective efficient lag in milliseconds for a batch

    Parameters
    ----------
    complex_pred : ndarray
                Batch of complex-valued predicted filtered analytic signal
    complex_gt : ndarray
                Batch of complex-valued ground truth analytic signal
    lag_shift : int
                Maximal shift in samples for cross-correlation
    sampling_rate : float
                Sampling rate of complex_pred and complex_gt
    dataset_name : str
                Name of the currently processed dataset

    Returns
    -------
    cross_correlation_values : dict
                Dictionary with keys: 'Correlation' which contains maximal correlation; 'Lag, ms' which contains
                respective lag value; 'Correlation2' which contains correlation value without shift
    """
    envelope_pred = np.abs(complex_pred)
    envelope_gt = np.abs(complex_gt)
    cross_correlation_values = {'Lag, ms': [], 'Correlation': [], 'Correlation2': []}
    if dataset_name in ['sines_white', 'sines_pink']:
        return cross_correlation_values
    for idx in range(complex_pred.shape[0]):
        pred = envelope_pred[idx, :]
        gt = envelope_gt[idx, :]
        assert pred.shape == gt.shape
        times4, corrs4 = crosscorr(gt, pred, sampling_rate, lag_shift)
        best_corr_idx4 = np.argmax(corrs4)
        corr4 = corrs4[best_corr_idx4]
        lag4 = 1000 * times4[best_corr_idx4]

        corrcoef = np.corrcoef(pred[lag_shift:-lag_shift], gt[lag_shift:-lag_shift])[0, 1]
        assert not (np.isnan(corr4) or np.isinf(corr4) or np.std(gt) < 1e-6), (corr4, np.std(gt), dataset_name)
        assert corr4 <= 1, (corr4, dataset_name)
        assert corr4 + 1e-6 >= corrcoef, (lag4 * 1000 / sampling_rate, dataset_name)
        cross_correlation_values['Correlation'].append(corr4)
        cross_correlation_values['Correlation2'].append(corrcoef)
        cross_correlation_values['Lag, ms'].append(lag4)
    return cross_correlation_values


def calculate_metrics(complex_pred: np.ndarray, complex_gt: np.ndarray, lag_shift: int, sampling_rate: float,
                      dataset_name: str):
    """
    Calculate correlation, respective true delay, Phase Locking Value (PLV), Circular standard deviation for a batch

    Parameters
    ----------
    complex_pred : ndarray
                Batch of complex-valued predicted filtered analytic signal
    complex_gt : ndarray
                Batch of complex-valued ground truth analytic signal
    lag_shift : int
                Maximal shift in samples for cross-correlation
    sampling_rate : float
                Sampling rate of complex_pred and complex_gt
    dataset_name : str
                Name of the currently processed dataset

    Returns
    -------
    detailed_errors_dict : dict
                Dictionary with keys: 'correlation', 'plv', 'circstd', 'circstd_degrees', 'delay' which contain lists
                with values of respective metrics for a batch
    """
    cross_correlation_values = calculate_batched_crosscorr(complex_pred=complex_pred,
                                                           complex_gt=complex_gt,
                                                           lag_shift=lag_shift,
                                                           sampling_rate=sampling_rate,
                                                           dataset_name=dataset_name)

    batch_size = complex_pred.shape[0]
    assert complex_gt.shape[0] == batch_size

    complex_pred = torch.tensor(complex_pred)
    complex_gt = torch.tensor(complex_gt)

    assert torch.is_complex(complex_pred)
    assert torch.is_complex(complex_gt)

    envelope_pred = torch.abs(complex_pred)
    envelope_gt = torch.abs(complex_gt)

    correlations_per_batch = calculate_batched_correlation(envelope_pred[:, lag_shift:-lag_shift],
                                                           envelope_gt[:, lag_shift:-lag_shift])
    plvs_per_batch = calculate_batched_plv(complex_pred, complex_gt)
    circstd_per_batch = calculate_batched_circstd(complex_pred, complex_gt)
    circstd_per_batch_degrees = circstd_per_batch * 180. / np.pi

    detailed_errors_dict = {
        'correlation': np.array(cross_correlation_values['Correlation']).tolist(),
        'plv': plvs_per_batch.cpu().numpy().tolist(),
        'circstd': circstd_per_batch.cpu().numpy().tolist(),
        'circstd_degrees': circstd_per_batch_degrees.cpu().numpy().tolist(),
        'delay': np.array(cross_correlation_values['Lag, ms']).tolist()
    }

    if dataset_name not in ['sines_white', 'sines_pink']:

        errors = correlations_per_batch.numpy()
        assert len(errors) == batch_size
        stats1 = {
            'min': np.min(errors),
            'max': np.max(errors),
            'mean': np.mean(errors),
            'median': np.median(errors)
        }
        errors = np.array(cross_correlation_values['Correlation'])
        stats2 = {
            'min': np.min(errors),
            'max': np.max(errors),
            'mean': np.mean(errors),
            'median': np.median(errors)
        }
        errors = np.array(cross_correlation_values['Correlation2'])
        stats3 = {
            'min': np.min(errors),
            'max': np.max(errors),
            'mean': np.mean(errors),
            'median': np.median(errors)
        }

        for stat_name in ['min', 'max', 'mean', 'median']:
            # stats and stats3 are just correlations, stats2 is the best correlation with delay, therefore
            # it must be bigger. 1e-6 for floats comparison
            assert stats2[stat_name] + 1e-6 >= stats1[stat_name], (stats2[stat_name], stats1[stat_name])
            assert stats2[stat_name] + 1e-6 >= stats3[stat_name], (stats2[stat_name], stats3[stat_name])

    return detailed_errors_dict


def test_statistical_results(metrics_to_calculate: list[str], save_path: str, csv_path: str, dataset_names: list[str],
                             model_type: str):
    """
    Compute results of statistical significance tests for difference of means in metrics

    Parameters
    ----------
    metrics_to_calculate : list
                A list of metrics to compute tests for. Available metrics: 'correlation', 'delay' and 'circstd_degrees'
    save_path : str
                A path to save a table with statistical significance test results
    csv_path : str
                A path to the csv file with results of experiments for base noise levels
    dataset_names : list
                Names of the datasets for which to perform tests
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks
    """
    no_correlation_simulations = ['sines_white', 'sines_pink']
    metrics = pd.read_csv(csv_path)
    algorithms_to_use = get_algorithms_list(model_type=model_type)
    if model_type == 'forecasting':
        base_model = 'Temporal Convolutional Network'
        if 'Delay, ms' in metrics.columns:
            metrics.loc[metrics['Algorithm'].isin(algorithms_to_use), 'Delay, ms'] -= 100.
    elif model_type == 'filtering':
        base_model = 'TCN filtering'
    else:
        raise ValueError('Only available model types are filtering and forecasting')
    metrics = metrics.loc[metrics['Simulation'].isin(dataset_names)]
    metrics = metrics.loc[metrics['Algorithm'].isin(algorithms_to_use)]
    if len(metrics.loc[~metrics['Simulation'].isin(no_correlation_simulations)]) == 0:
        metrics_to_calculate = ['circstd_degrees']
    table = {'Base algorithm': [], 'Algorithm': [], 'Simulation': [], 'Metric': [],
             'Base algorithm mean': [], 'Algorithm mean': [],
             'Mann-Whitney U two-sided rank test p-value': [],
             'Mann-Whitney U two-sided rank test statistic': []
             }
    for simulation in dataset_names:
        df = metrics.loc[metrics['Simulation'] == simulation]
        for metric_id in metrics_to_calculate:
            full_metric_name = get_full_metric_name(metric_id=metric_id, visualization_mode=False)
            df_metric = df[['Algorithm', 'Simulation', full_metric_name]]
            if metric_id != 'circstd_degrees':
                df_metric = df_metric.loc[~df_metric['Simulation'].isin(no_correlation_simulations)]
            tcn_results = df_metric.loc[df_metric['Algorithm'] == base_model][full_metric_name].to_numpy()
            assert tcn_results.ndim == 1

            for algorithm in algorithms_to_use:
                if algorithm == base_model:
                    continue
                algorithm_results = df_metric.loc[df_metric['Algorithm'] == algorithm][full_metric_name].to_numpy()
                assert algorithm_results.ndim == 1
                assert tcn_results.shape == algorithm_results.shape

                t_2_sided_test = stats.mannwhitneyu(x=tcn_results, y=algorithm_results, alternative='two-sided',
                                                    method='exact')
                t_2_sided_stat = t_2_sided_test.statistic
                t_2_sided_p = t_2_sided_test.pvalue

                table['Base algorithm'].append(base_model)
                table['Algorithm'].append(algorithm)
                table['Simulation'].append(simulation)
                table['Metric'].append(full_metric_name)
                table['Base algorithm mean'].append(np.mean(tcn_results))
                table['Algorithm mean'].append(np.mean(algorithm_results))
                table['Mann-Whitney U two-sided rank test p-value'].append(t_2_sided_p)
                table['Mann-Whitney U two-sided rank test statistic'].append(t_2_sided_stat)

    table = pd.DataFrame(table)
    create_parents(filepath=save_path)
    table.to_csv(save_path, index=False)


def get_main_plotting_csv(filepaths: list[str], metrics_to_calculate: list[str], dataset_names: list[str],
                          lag_shift: int, save_path: tp.Optional[str]):
    """
    Return and save (optionally) a DataFrame with results of experiments for base noise levels

    Parameters
    ----------
    filepaths : list
                Paths to json files with test metrics which should be included into the DataFrame
    metrics_to_calculate : list
                A list of metrics to compute tests for. Available metrics: 'correlation', 'delay' and 'circstd_degrees'
    dataset_names : list
                Names of the datasets to include into a DataFrame
    lag_shift : int
                Maximal shift in samples for cross-correlation
    save_path : str or None
                A path to save a DataFrame with results of experiments for base noise levels. If None, will not save

    Returns
    -------
    metrics : DataFrame
                DataFrame with the following columns: 'Algorithm' is a name of an algorithm used; 'Simulation' is a
                name of a dataset for which metrics are calculated; columns with metric values for metrics_to_calculate
    """
    test_dataset_names(dataset_names=dataset_names, use_synthetic=True, use_real=True)
    metrics = {get_full_metric_name(metric_id=metric_id, visualization_mode=False): [] for metric_id in
               metrics_to_calculate}
    metrics['Algorithm'] = []
    metrics['Simulation'] = []
    processed_experiments = set()
    for filepath in tqdm(filepaths, desc=f'Computing main csv for {", ".join(dataset_names)}'):
        test_jsf = read_json(filepath=filepath)
        dataset_name = test_jsf['dataset_name']
        sampling_rate = test_jsf['sampling_rate']
        assert sampling_rate == 250
        if dataset_name not in dataset_names:
            continue
        # For synthetic datasets only noise_coeff == 1.0 is used in the main graphs.
        # Other noise_coeffs are required for the noise robustness graphs which are
        # prepared by get_noise_plotting_csv.
        if (dataset_name != 'multiperson_real_data') and (abs(test_jsf['noise_coeff'] - 1.) >= 1e-9):
            continue

        if dataset_name == 'multiperson_real_data':
            npz_filepaths = get_real_data_npz_filepaths(filepath=filepath)
            complex_pred, complex_gt = extract_multiperson_predictions_gt(filepaths=npz_filepaths)
        else:
            complex_pred, complex_gt = extract_predictions_gt(filepath=filepath)
        model_name = get_model_name(get_model_id(filepath=filepath))

        processing_experiment = (dataset_name, model_name)
        assert processing_experiment not in processed_experiments, f'{dataset_name} is presented multiple times with ' \
                                                                   f'the same noise_coeff == 1.0 in filepaths for ' \
                                                                   f'the same model {model_name}. That is not allowed' \
                                                                   f' as it will lead to mixing results from ' \
                                                                   f'different experiments'
        processed_experiments.add(processing_experiment)

        errors_size = len(complex_pred)
        detailed_errors_dict = calculate_metrics(complex_pred=complex_pred,
                                                 complex_gt=complex_gt,
                                                 lag_shift=lag_shift,
                                                 sampling_rate=sampling_rate,
                                                 dataset_name=dataset_name)
        for metric_id in metrics_to_calculate:
            full_metric_name = get_full_metric_name(metric_id=metric_id, visualization_mode=False)
            if (metric_id != 'circstd_degrees') and (dataset_name in ['sines_white', 'sines_pink']):
                # Sines have constant amplitude, so metrics based on correlation of amplitudes are not applicable
                metrics[full_metric_name].extend([np.nan] * errors_size)
                continue
            error_values = detailed_errors_dict[metric_id]
            assert len(error_values) == errors_size
            metrics[full_metric_name].extend(error_values)
        metrics['Algorithm'].extend([model_name] * errors_size)
        metrics['Simulation'].extend([dataset_name] * errors_size)
    metrics = pd.DataFrame(metrics)
    order = {'Sines white': 0,
             'Sines pink': 1,
             'Filtered pink': 2,
             'State space white': 3,
             'State space pink': 4,
             'Real EEG data': 5}
    metrics = metrics.sort_values(by=['Simulation'], key=lambda x: x.map(order))
    if save_path is not None:
        create_parents(filepath=save_path)
        metrics.to_csv(save_path, index=False)
    return metrics


def get_noise_plotting_csv(filepaths: list[str], metrics_to_calculate: list[str], dataset_name: str,
                           lag_shift: int, save_path: tp.Optional[str]):
    """
    Return and save (optionally) a DataFrame with results of experiments for varying noise levels

    Parameters
    ----------
    filepaths : list
                Paths to json files with test metrics which should be included into the DataFrame
    metrics_to_calculate : list
                A list of metrics to compute tests for. Available metrics: 'correlation', 'delay' and 'circstd_degrees'
    dataset_name : str
                Name of a dataset to compute metrics for. Should be one of 'filtered_pink' or 'state_space_pink'
    lag_shift : int
                Maximal shift in samples for cross-correlation
    save_path : str or None
                A path to save a DataFrame with results of experiments for varying noise levels. If None, will not save

    Returns
    -------
    metrics : DataFrame
                DataFrame with the following columns: 'Algorithm' is a name of an algorithm used; 'Simulation' is a
                name of a dataset for which metrics are calculated; 'Noise level' with a calculated noise level (NL)
                from noise coefficient; columns with metric values for metrics_to_calculate
    """
    assert dataset_name in ['filtered_pink', 'state_space_pink']
    metrics = {get_full_metric_name(metric_id=metric_id, visualization_mode=False): [] for metric_id in
               metrics_to_calculate}
    metrics['Algorithm'] = []
    metrics['Simulation'] = []
    metrics['Noise level'] = []
    processed_experiments = set()
    for filepath in tqdm(filepaths, desc=f'Computing noise csv for {dataset_name}'):
        test_jsf = read_json(filepath=filepath)
        current_dataset_name = test_jsf['dataset_name']
        sampling_rate = test_jsf['sampling_rate']
        assert sampling_rate == 250
        if current_dataset_name != dataset_name:
            continue
        noise_coeff = test_jsf['noise_coeff']

        model_name = get_model_name(get_model_id(filepath=filepath))

        processing_experiment = (dataset_name, model_name, noise_coeff)
        assert processing_experiment not in processed_experiments, f'{dataset_name} is presented multiple ' \
                                                                   f'times with the same noise_coeff {noise_coeff} ' \
                                                                   f'in filepaths for the same model {model_name}. ' \
                                                                   f'That is not allowed as it will lead to mixing ' \
                                                                   f'results from different experiments'
        processed_experiments.add(processing_experiment)

        complex_pred, complex_gt = extract_predictions_gt(filepath)
        errors_size = len(complex_pred)
        detailed_errors_dict = calculate_metrics(complex_pred=complex_pred,
                                                 complex_gt=complex_gt,
                                                 lag_shift=lag_shift,
                                                 sampling_rate=sampling_rate,
                                                 dataset_name=current_dataset_name)
        for metric_id in metrics_to_calculate:
            full_metric_name = get_full_metric_name(metric_id=metric_id, visualization_mode=False)
            error_values = detailed_errors_dict[metric_id]
            assert len(error_values) == errors_size
            metrics[full_metric_name].extend(error_values)
        metrics['Algorithm'].extend([model_name] * errors_size)
        metrics['Simulation'].extend([current_dataset_name] * errors_size)
        # Noise level (NL) from the paper == 10 * noise_coeff for the pink noise
        metrics['Noise level'].extend([10 * noise_coeff] * errors_size)
    metrics = pd.DataFrame(metrics)
    if save_path is not None:
        create_parents(filepath=save_path)
        metrics.to_csv(save_path, index=False)
    return metrics


def get_csv_path(save_name: str):
    """
    Get a path to a csv file in RESULTS_ROOT/visualization_csv_files directory

    Parameters
    ----------
    save_name : str
                Name of a csv file

    Returns
    -------
    csv_path : str
                Path to a csv file
    """
    return os.path.join(RESULTS_ROOT, 'visualization_csv_files', f'{save_name}.csv')


def get_experiment_parameters(experiment_name: str, model_type: str, return_paths: bool):
    """
    Get dataset names; metrics to calculate; save names (or paths) of csv file with main experiments for base noise
    level, for varying noise level experiments (when applicable), for statistical testing; dataset name for varying
    noise level experiments (when applicable)

    Parameters
    ----------
    experiment_name : str
                Experiment name to plot. One of 'sines', 'filtered_pink', 'state_space' and 'multiperson_real_data'
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks
    return_paths : bool
                If True will return paths to csv files instead of names

    Returns
    -------
    dataset_names : list
                Names of the datasets to compute metrics for base noise levels
    metrics_to_calculate : list
                A list of metrics to compute tests for. Available metrics: 'correlation', 'delay' and 'circstd_degrees'
    save_name : str
                A name (or path) to csv with results of experiments for base noise levels
    stats_save_name : str
                A name (or path) to csv with results of statistical significance tests
    noise_dataset_name : str or None
                Name of a dataset to compute metrics with varying noise levels for. Will be 'filtered_pink' for
                experiment_name == 'filtered_pink', 'state_space_pink' for experiment_name == 'state_space', None
                otherwise
    noise_save_name : str or None
                A name (or path) to csv with results of experiments for varying noise levels. If experiment_name not
                'filtered_pink' or 'state_space_pink' returns None
    """
    metrics_to_calculate = ['correlation', 'delay', 'circstd_degrees']
    noise_save_name = None
    noise_dataset_name = None
    if experiment_name == 'sines':
        dataset_names = ['sines_white', 'sines_pink']
        save_name = f'{model_type}_sines'
        metrics_to_calculate = ['circstd_degrees']
    elif experiment_name == 'filtered_pink':
        dataset_names = ['filtered_pink']
        save_name = f'{model_type}_filtered_pink'
        noise_save_name = f'noise_{save_name}'
        noise_dataset_name = 'filtered_pink'
    elif experiment_name == 'state_space':
        dataset_names = ['state_space_white', 'state_space_pink']
        save_name = f'{model_type}_state_space'
        noise_save_name = f'noise_{save_name}_pink'
        noise_dataset_name = 'state_space_pink'
    elif experiment_name == 'multiperson_real_data':
        assert model_type == 'filtering'
        dataset_names = ['multiperson_real_data']
        save_name = f'{model_type}_multiperson_real_data'
    else:
        raise ValueError('Choose valid experiment name from [sines, filtered_pink, state_space, multiperson_real_data]')
    stats_save_name = f'stats_{save_name}'
    if return_paths:
        save_name = get_csv_path(save_name=save_name)
        stats_save_name = get_csv_path(save_name=stats_save_name)
        if noise_save_name is not None:
            noise_save_name = get_csv_path(save_name=noise_save_name)
    return dataset_names, metrics_to_calculate, save_name, stats_save_name, noise_dataset_name, noise_save_name


def create_csv_files_for_plots(visualization_filepaths: str, model_type: str):
    """
    For 'sines', 'filtered_pink', 'state_space' (and 'multiperson_real_data' if model_type == 'filtering') experiments
    creates csv tables for plotting

    Parameters
    ----------
    visualization_filepaths : str
                A path to a txt file with paths to json files with test results to consider
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks
    """
    assert model_type in ['forecasting', 'filtering']
    with open(visualization_filepaths) as vis_paths:
        filepaths = []
        for line in vis_paths:
            line = line.rstrip()
            if line != '':
                filepaths.append(line)

    lag_shift = 100
    experiment_names = ['sines', 'filtered_pink', 'state_space']

    if model_type == 'filtering':
        experiment_names.append('multiperson_real_data')

    for experiment_name in experiment_names:
        dataset_names, metrics_to_calculate, main_save_path, stats_save_path, noise_dataset_name, \
            noise_save_path = get_experiment_parameters(experiment_name=experiment_name, model_type=model_type,
                                                        return_paths=True)
        metrics_csv = get_main_plotting_csv(filepaths=filepaths, metrics_to_calculate=metrics_to_calculate,
                                            dataset_names=dataset_names,
                                            lag_shift=lag_shift, save_path=main_save_path)
        if noise_save_path is not None:
            get_noise_plotting_csv(filepaths=filepaths, metrics_to_calculate=metrics_to_calculate,
                                   dataset_name=noise_dataset_name, lag_shift=lag_shift, save_path=noise_save_path)

        test_statistical_results(metrics_to_calculate=metrics_to_calculate, save_path=stats_save_path,
                                 csv_path=main_save_path,
                                 dataset_names=dataset_names, model_type=model_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating supplementary csv files needed for plotting.')
    parser.add_argument('-p', '--path', type=str, help='Path to the txt file with test json files for specified '
                                                       'model_type of experiments which are supposed to be used in '
                                                       'plotting.'
                                                       'Example of one of such paths: '
                                                       'results/test_results/TCNFilteringModel/YYYY-MM-DD/00-00-00/test_metrics_0.json'
                                                       'if you run this file from the root',
                        required=True)
    parser.add_argument('-m', '--model_type', type=str,
                        choices=['forecasting', 'filtering'],
                        help='Name of the model type for which the txt file is created. '
                             'Note: you should pass separately forecasting and filtering models in different txt files',
                        required=True)
    args = parser.parse_args()
    create_csv_files_for_plots(visualization_filepaths=args.path, model_type=args.model_type)

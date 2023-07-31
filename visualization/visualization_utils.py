import os
import numpy as np
from general_utilities.constants import RESULTS_ROOT


def extract_relative_path(filepath: str):
    """
    Returns relative to test_results filepath without extension

    Parameters
    ----------
    filepath : str
                A path to something in test_results directory or its subdirectories

    Returns
    -------
    rel_path : str
                A relative to test_results filepath without extension
    """
    return os.path.splitext(os.path.relpath(filepath, os.path.join('results', 'test_results')))[0]


def get_model_id(filepath: str):
    """
    Returns model id from filepath in the directory of its experiment. It assumes the following structure of the
    filepath SOME-ROOT/model_id/subdir/subdir/file. Useful to extract model_id from paths to model's results like
    RESULTS_ROOT/test_results/model_id/YYYY-MM-DD/00-00-00/test_metrics_0.json

    Parameters
    ----------
    filepath : str
                A path with a structure SOME-ROOT/model_id/subdir/subdir/file

    Returns
    -------
    model_id : str
                Model id
    """
    for _ in range(3):
        filepath = os.path.dirname(filepath)
    model_id = os.path.basename(filepath)
    return model_id


def extract_predictions_gt(filepath: str):
    """
    Extracts saved complex-valued predicted filtered and ground truth analytic signal

    Parameters
    ----------
    filepath : str
                A path to test_metrics_{idx}.json for a model

    Returns
    -------
    complex_pred : ndarray
                Batch of complex-valued predicted filtered analytic signal
    complex_gt : ndarray
                Batch of complex-valued ground truth analytic signal
    """
    npz_filepath = os.path.join(RESULTS_ROOT, 'saved_predictions', extract_relative_path(filepath=filepath) + '.npz')
    data = np.load(npz_filepath)
    complex_pred = data['complex_pred']
    complex_gt = data['complex_gt']
    return complex_pred, complex_gt


def extract_multiperson_predictions_gt(filepaths: list[str]):
    """
    Extracts saved complex-valued predicted filtered and ground truth analytic signal for multiple people

    Parameters
    ----------
    filepaths : list
                A paths to npz files with saved predictions and ground truth for different people

    Returns
    -------
    complex_preds : ndarray
                Batch of concatenated complex-valued predicted filtered analytic signal for multiple people
    complex_gts : ndarray
                Batch of concatenated complex-valued ground truth analytic signal for multiple people
    """
    complex_preds = []
    complex_gts = []
    for filepath in filepaths:
        data = np.load(filepath)
        complex_pred = data['complex_pred']
        complex_gt = data['complex_gt']
        complex_preds.append(complex_pred)
        complex_gts.append(complex_gt)
    complex_preds = np.concatenate(complex_preds)
    complex_gts = np.concatenate(complex_gts)
    return complex_preds, complex_gts


def get_real_data_npz_filepaths(filepath: str):
    """
    Extracts paths to npz files with saved predictions and ground truth for different people

    Parameters
    ----------
    filepath : str
                A path to test_metrics_{idx}.json for a model

    Returns
    -------
    filepaths : list
                A paths to npz files with saved predictions and ground truth for different people
    """
    people_ids = [0, 11, 22, 30, 49]
    npz_filepath = os.path.join(RESULTS_ROOT, 'saved_predictions', extract_relative_path(filepath) + '.npz')
    npz_filepath = npz_filepath.split('_')
    root = '_'.join(npz_filepath[:-1])
    fin = npz_filepath[-1]
    filepaths = []
    for person_idx in people_ids:
        npz_path = f'{root}_person_{person_idx}_{fin}'
        filepaths.append(npz_path)
    return filepaths


def get_algorithms_list(model_type: str):
    """
    Returns a list of considered algorithms for a model_type

    Parameters
    ----------
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks

    Returns
    -------
    algorithms_to_use : list
                A list of considered algorithms for a model_type
    """
    if model_type == 'forecasting':
        algorithms_to_use = ['DLinear', 'N-HiTS', 'NLinear', 'Temporal Convolutional Network', 'Transformer Network']
    elif model_type == 'filtering':
        algorithms_to_use = ['cFIR', 'Kalman Filter', 'TCN filtering', 'Conv-TasNet filtering']
    else:
        raise ValueError('Only available model types are filtering and forecasting')
    return algorithms_to_use


def get_simulation_names():
    """
    Get a dictionary of mappings of dataset_names to name for visualization

    Returns
    -------
    simulation_names : dict
                A dictionary of mappings of dataset_names to name for visualization
    """
    simulation_names = {
        'sines_white': 'Sines\nwhite',
        'sines_pink': 'Sines\npink',
        'filtered_pink': 'Filtered\npink',
        'state_space_white': 'STSP\nwhite',
        'state_space_pink': 'STSP\npink',
        'multiperson_real_data': 'Real EEG data',
    }
    return simulation_names


def get_simulation_name(dataset_name: str):
    """
    Get a name for visualization for a dataset_name

    Parameters
    ----------
    dataset_name : str
                A name of a dataset to visualize

    Returns
    -------
    simulation_name : str
                A dataset name for visualization
    """
    simulation_names = get_simulation_names()
    return simulation_names[dataset_name]


def get_model_name(model_id: str):
    """
    Get a name for visualization for a model_id

    Parameters
    ----------
    model_id : str
                An id of a model to visualize

    Returns
    -------
    model_name : str
                A model name for visualization
    """
    models_rename = {'TCNFilteringModel': 'TCN filtering',
                     'ConvTasNetFilteringModel': 'Conv-TasNet filtering',
                     'DLinearModel': 'DLinear',
                     'NHiTSModel': 'N-HiTS',
                     'NLinearModel': 'NLinear',
                     'TCNModel': 'Temporal Convolutional Network',
                     'TransformerModel': 'Transformer Network',
                     'cFIR': 'cFIR',
                     'KalmanFilter': 'Kalman Filter'}
    return models_rename[model_id]


def get_full_metric_name(metric_id: str, visualization_mode: bool = False):
    """
    Get a full name of a metric or its name for visualization

    Parameters
    ----------
    metric_id : str
                An id of a metric
    visualization_mode : bool
                If True will return shorter name suitable for visualization, else returns full metric name

    Returns
    -------
    metric_name : str
                A full metric name or metric name for visualization
    """
    metric_names = {'circstd_degrees': 'Circular standard deviation, degrees',
                    'correlation': 'Envelopes correlation',
                    'delay': 'Delay, ms'}
    metric_vis_name = {'circstd_degrees': 'Circular STD, degrees',
                       'correlation': 'Envelopes correlation',
                       'delay': 'Delay, ms'}
    if visualization_mode:
        metric_name = metric_vis_name[metric_id]
    else:
        metric_name = metric_names[metric_id]
    return metric_name


def get_margins(mode: str, pdf_dpi: float = 100):
    """
    Get sizes of a single metrics subfigure and a layout of subfigures

    Parameters
    ----------
    mode : str
                For which experiment sizes are required. Should be either main (for main experiments with base noise
                level) or noise (for experiments with varying noise levels)
    pdf_dpi : float
                DPI for a figure

    Returns
    -------
    dx : float
                Size for the x axis
    dy : float
                Size for the y axis
    layout : dict
                Layout of subfigures which can be split
    """

    if mode == 'main':
        dx = 350 / pdf_dpi
        dy = 212 / pdf_dpi
        layout = {
            'sines': {'circstd_degrees': [[0., 0.], [dx + 0.5, dy + 0.1]],
                      'legend': [[dx + 0.6, 0.15], [2 * dx - 0.45, dy + 0.25]]},
            'all_metrics': {'circstd_degrees': [[0., 0.], [dx, dy]],
                            'correlation': [[0., dy], [dx, 2 * dy]],
                            'delay': [[dx, dy], [2 * dx, 2 * dy]],
                            'legend': [[dx, 0.], [2 * dx, dy]]}
        }
    elif mode == 'noise':
        dx = 560 / pdf_dpi
        dy = 400 / pdf_dpi
        layout = {
            'all_metrics': [[0, 2 * dy / 3 + 1.], [dx, 2 * dy]],
            'legend': [[1.4, 1.], [dx - 1., 2 * dy / 3]]
        }
    else:
        raise ValueError(f'Mode: {mode} not recognized. Use main for main graphs and noise for robustness graphs.')
    return dx, dy, layout


def test_dataset_names(dataset_names: list, use_synthetic: bool, use_real: bool):
    """
    Tests if dataset_names are correct

    Parameters
    ----------
    dataset_names : list
                Names of datasets to check
    use_synthetic : bool
                If dataset_names should include synthetic datasets
    use_real : bool
                If dataset_names should include real dataset
    """
    available_dataset_names = []
    if use_synthetic:
        available_dataset_names = available_dataset_names + ['sines_white', 'sines_pink', 'filtered_pink',
                                                             'state_space_white', 'state_space_pink']
    if use_real:
        available_dataset_names = available_dataset_names + ['multiperson_real_data']
    for dataset_name in dataset_names:
        assert dataset_name in available_dataset_names, f'dataset_name: {dataset_name} is not in the ' \
                                                        f'available dataset names {available_dataset_names} for this ' \
                                                        f'experiment'

import argparse
import os
from glob import glob
from omegaconf import OmegaConf
from general_utilities.constants import PROJECT_ROOT, RESULTS_ROOT


def get_new_line(experiment_path: str, dataset_name: str, model_type: str):
    """
    Generate new line for bash script

    Parameters
    ----------
    experiment_path : str
                Path to the experiment which has trained model.
                Example: RESULTS_ROOT/outputs/TCNModel/YYYY-MM-DD/00-00-00/
    dataset_name : str
                Dataset name. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                'state_space_white', 'state_space_pink', 'multiperson_real_data']
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks

    Returns
    -------
    line : str
                Line with testing command
    """
    line = ['python']
    if dataset_name == 'multiperson_real_data':
        assert model_type == 'filtering'
        script_name = 'eeg_nn_filtering/test_filtering_model_real.py'
        ds = '--dataset_path "${DATASET_PATH}"'
    elif model_type == 'filtering':
        script_name = 'eeg_nn_filtering/test_filtering_model_synthetic.py'
        ds = f'--dataset_name {dataset_name}'
    elif model_type == 'forecasting':
        script_name = 'eeg_forecast/test_forecasting_model.py'
        ds = f'--dataset_name {dataset_name}'
    else:
        raise ValueError
    line.append(script_name)
    line.append(ds)
    path_part = f'--path {os.path.relpath(experiment_path, PROJECT_ROOT)}'
    line.append(path_part)
    return ' '.join(line)


def get_all_test_paths(models: list[str], dates: list[str], model_type: str):
    """
    Generate commands for testing bash script

    Parameters
    ----------
    models : list
                List of models to consider
    dates : list
                List of dates to consider
    model_type : str
                Neural network type. Should be 'filtering' for EEG filtration networks and 'forecasting' for
                forecasting darts networks
    """
    known_dataset_names = {
        'sines_white_10Hz_250Hz.npz': 'sines_white',
        'sines_pink_10Hz_250Hz.npz': 'sines_pink',
        'pink_filt_8_12_250Hz.npz': 'filtered_pink',
        'state_space_white_10Hz_250Hz.npz': 'state_space_white',
        'state_space_pink_10Hz_250Hz.npz': 'state_space_pink',
        'real_data_250Hz.npz': 'multiperson_real_data'
    }
    ordering = ['sines_white', 'sines_pink', 'filtered_pink', 'state_space_white', 'state_space_pink']
    priorities = {dataset_name: idx for idx, dataset_name in enumerate(ordering)}
    real_lines = []
    synthetic_lines = []
    for model_id in models:
        synthetic_lines.append(f'# {model_id} Synthetic')
        real_lines.append(f'# {model_id} Real')
        per_model_synthetic_lines = []
        for experiment_date in dates:
            for experiment_path in glob(os.path.join(RESULTS_ROOT, 'outputs', model_id, experiment_date, '*')):
                cfg = OmegaConf.load(os.path.join(experiment_path, '.hydra', 'config.yaml'))
                dataset_name = known_dataset_names[os.path.basename(cfg['dataset_relative_path'])]
                line = get_new_line(experiment_path=experiment_path, dataset_name=dataset_name, model_type=model_type)
                if dataset_name == 'multiperson_real_data':
                    real_lines.append(line)
                else:
                    per_model_synthetic_lines.append((line, priorities[dataset_name]))
        per_model_synthetic_lines = sorted(per_model_synthetic_lines, key=lambda x: x[1])
        synthetic_lines = synthetic_lines + [line for line, _ in per_model_synthetic_lines]
        synthetic_lines.append('\n')
        real_lines.append('\n')
    all_lines = '\n'.join(synthetic_lines + real_lines)
    print(all_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output formatted paths for testing scripts.')
    parser.add_argument('-m', '--models', type=str, nargs='+', help='Names of models to use.',
                        required=True)
    parser.add_argument('-d', '--dates', type=str, nargs='+',
                        help='Dates of experiments to check for models.', required=True)
    parser.add_argument('-t', '--model_type', type=str, choices=['filtering', 'forecasting'],
                        help='Model type.', required=True)
    args = parser.parse_args()
    get_all_test_paths(models=args.models, dates=args.dates, model_type=args.model_type)

import json
import random
import warnings
import numpy as np
import torch
from omegaconf import open_dict


def read_json(filepath: str):
    """
    Read json file

    Parameters
    ----------
    filepath : str
                Path to json file

    Returns
    -------
    jsonfile : dict
                Data of the json file
    """
    with open(filepath) as f:
        jsonfile = json.load(f)
    return jsonfile


def save_json(data: dict, filepath: str):
    """
    Save json file

    Parameters
    ----------
    data : dict
                Data of the json file
    filepath : str
                Path to json file

    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def calculate_statistics(data: torch.Tensor, required_statistics: list[str]):
    """
    Calculate aggregated statistics

    Parameters
    ----------
    data : torch.Tensor
                Tensor with data
    required_statistics : list
                List of aggregated statistics to compute. Choose from ['min', 'max', 'mean', 'median']

    Returns
    -------
    aggregated_stats : dict
                Dictionary with aggregated statistics
    """
    statistics_functions = {'min': torch.min,
                            'max': torch.max,
                            'mean': torch.mean,
                            'median': torch.median}
    results = {}
    for stat_name in required_statistics:
        results[stat_name] = statistics_functions[stat_name](data).item()
    return results


def fill_specific_params_dict(common_params_dict: dict, specific_params_dict: dict):
    """
    Add inplace all keys and values from common_params_dict to specific_params_dict

    Parameters
    ----------
    common_params_dict : dict
                Dictionary with parameters
    specific_params_dict : dict
                Dictionary with parameters
    """
    with open_dict(specific_params_dict):
        for param, param_value in common_params_dict.items():
            specific_params_dict[param] = param_value


def get_devices(preferred_device: str):
    """
    Get PyTorch device and Pytorch Lightning accelerator and devices for Trainer

    Parameters
    ----------
    preferred_device : str
                Preferred device as 'cpu' or 'gpu'. If 'gpu' chosen, but cuda is not available for PyTorch will use
                'cpu'

    Returns
    -------
    device : str
                PyTorch device name
    trainer_accelerator : str
                Pytorch Lightning Trainer accelerator name
    trainer_devices : list or str
                Pytorch Lightning Trainer devices list/mode
    """
    assert preferred_device in ['cpu', 'gpu']
    if preferred_device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('Preferred device is GPU, but CUDA is not available. Computations will continue on CPU')
        preferred_device = 'cpu'
    if preferred_device == 'cpu':
        device = 'cpu'
        trainer_accelerator = 'cpu'
        trainer_devices = 'auto'
    elif preferred_device == 'gpu':
        device = 'cuda'
        trainer_accelerator = 'gpu'
        trainer_devices = [0]
    else:
        raise ValueError
    return device, trainer_accelerator, trainer_devices


def make_deterministic(seed: int):
    """
    Make random operations deterministic

    Parameters
    ----------
    seed : int
                Seed for random generators
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

import typing as tp
from functools import partial
import optuna
from eeg_classic_filtering.datasets import get_multiperson_datasets, get_synthetic_datasets
from eeg_classic_filtering.helpful_functions import save_multipers_results


def get_generator_parameters(dataset_name: str):
    """
    Returns appropriate parameters: central frequency, filter band and noise coefficients for the dataset

    Parameters
    ----------
    dataset_name : str
                Name of the dataset. One of ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink',
                'filtered_pink']

    Returns
    -------
    frequency : int or None
                Central frequency for 'sines_white', 'sines_pink', 'state_space_white', 'state_space_pink' datasets.
                None for 'filtered_pink' as it relies on filter band
    data_filter_band : tuple (band_low, band_high) or None
                Filtering band to generate 'filtered_pink' dataset. None for others
    noise_coeffs : list
                Noise coefficients for test datasets generation. Is used to test robustness to pink noise in
                'filtered_pink' and 'state_space_pink' experiments
    """
    if dataset_name in ['sines_white', 'sines_pink', 'state_space_white', 'state_space_pink']:
        frequency = 10
        data_filter_band = None
    elif dataset_name == 'filtered_pink':
        frequency = None
        data_filter_band = (8, 12)
    else:
        raise ValueError('dataset_name is not recognized. Please choose one from the following list of synthetic '
                         'datasets: [sines_white, sines_pink, filtered_pink, state_space_white, state_space_pink].')

    if dataset_name in ['sines_white', 'sines_pink', 'state_space_white']:
        noise_coeffs = [1.0]
    elif dataset_name in ['filtered_pink', 'state_space_pink']:
        noise_coeffs = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
    else:
        raise ValueError('dataset_name is not recognized. Please choose one from the following list of synthetic '
                         'datasets: [sines_white, sines_pink, filtered_pink, state_space_white, state_space_pink].')
    return frequency, data_filter_band, noise_coeffs


def get_best_params(metric_name: str, optuna_additional_params: dict, optuna_n_trials: int,
                    optuna_objective: tp.Callable, optuna_parameters_restructurer: tp.Optional[tp.Callable] = None):
    """
    Find the best parameters with optuna framework according to the specified metric for the algorithm

    Parameters
    ----------
    metric_name : str
                Name of the metric for optimization. One of ['correlation', 'plv', 'circstd', 'circstd_degrees']
    optuna_additional_params : dict
                Arguments required by optuna_objective aside from trial
    optuna_n_trials : int
                Amount of trials for optimization
    optuna_objective : Callable
                Objective function which returns appropriate metric value
    optuna_parameters_restructurer : Callable or None
                A function which restructures a dictionary with model parameters. If not needed to restructure use None

    Returns
    -------
    best_params : dict
                A dictionary with best found parameters
    """
    optuna_additional_params['metric_name'] = metric_name
    partial_optuna_objective = partial(optuna_objective,
                                       **optuna_additional_params)
    if metric_name in ['correlation', 'plv']:
        direction = 'maximize'
    elif metric_name in ['circstd', 'circstd_degrees']:
        direction = 'minimize'
    else:
        raise ValueError('metric_name is not recognized. Please choose one from the following list of '
                         'metrics: [correlation, plv, circstd, circstd_degrees]. Note: correlation should not be '
                         'used for sines_white and sines_pink datasets because sines have constant amplitude and '
                         'therefore constant series are not appropriate to use for correlation.')
    sampler = optuna.samplers.TPESampler(seed=10)  # Deterministic optimization
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(partial_optuna_objective, n_trials=optuna_n_trials, n_jobs=1)
    best_params = study.best_params
    if optuna_parameters_restructurer is not None:
        best_params = optuna_parameters_restructurer(best_params)

    print(best_params)

    return best_params


def optimize_and_test_synthetic(model_name: str, optuna_objective: tp.Callable,
                                optuna_parameters_restructurer: tp.Optional[tp.Callable],
                                dataset_name: str, train_dataset_path: str, add_X_dimension: bool,
                                test_function: tp.Callable, n_trials: int = 100,
                                train_amount_of_observations: int = 4500):
    """
    Find the best parameters with optuna framework for the algorithm. Use them on the generated test dataset and save
    results

    Parameters
    ----------
    model_name : str
                Name of the model. Typically used by a test_function to write to appropriate model folder in the
                results directory results/test_results/model_name/YYYY-MM-DD/00-00-00
    optuna_objective : Callable
                Objective function which returns appropriate metric value
    optuna_parameters_restructurer : Callable or None
                A function which restructures a dictionary with model parameters. If not needed to restructure use None
    dataset_name : str
                Name of the dataset to test. One of ['sines_white', 'sines_pink',
                'state_space_white', 'state_space_pink', 'filtered_pink']
    train_dataset_path : str
                Path to the training pre-split synthetic dataset
    add_X_dimension : bool
                Whether to add additional dimension to the input time-series. If False input time-series shape
                is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications
    test_function : Callable
                A function used to run a single test for a specific hyperparameters of a filter and calculate metrics
    n_trials : int
                Amount of trials for optimization
    train_amount_of_observations : int
                Amount of observations for optimization of parameters
    """
    metric_names = {
        'sines_white': 'circstd_degrees',
        'sines_pink': 'circstd_degrees',
        'filtered_pink': 'correlation',
        'state_space_white': 'correlation',
        'state_space_pink': 'correlation'
    }

    test_sampling_rate = 250
    best_params_for_dataset = None
    test_frequency, test_data_filter_band, noise_coeffs = get_generator_parameters(dataset_name=dataset_name)

    for noise_coeff in noise_coeffs:

        train_ds, test_ds = get_synthetic_datasets(train_dataset_path=train_dataset_path,
                                                   train_gt_key='HFIR_GT',
                                                   train_first_observation_idx=0,
                                                   train_amount_of_observations=train_amount_of_observations,
                                                   test_dataset_name=dataset_name,
                                                   test_frequency=test_frequency, test_duration=20,
                                                   test_sampling_rate=test_sampling_rate,
                                                   test_data_filter_band=test_data_filter_band,
                                                   test_amount_of_observations=180,
                                                   test_noise_coeff=noise_coeff,
                                                   add_X_dimension=add_X_dimension)

        if best_params_for_dataset is None:
            # We want to optimize on noise_coeff = 1.0 and test on other noise_coeff without retuning.
            # It must go first in these lists. But assert is provided in case.
            assert abs(noise_coeff - 1.0) < 1e-9  # comparing floats
            metric_name = metric_names[dataset_name]
            best_params = get_best_params(metric_name=metric_name,
                                          optuna_additional_params={
                                              'sampling_rate': test_sampling_rate,
                                              'dataset': train_ds,
                                              'dataset_type': 'pre_split',
                                              'dataset_name': dataset_name
                                          },
                                          optuna_n_trials=n_trials,
                                          optuna_objective=optuna_objective,
                                          optuna_parameters_restructurer=optuna_parameters_restructurer)
            best_params_for_dataset = best_params
        else:
            best_params = best_params_for_dataset

        test_function(sampling_rate=test_sampling_rate,
                      dataset=test_ds,
                      dataset_type='generated',
                      dataset_name=dataset_name,
                      frequency=test_frequency,
                      filter_band=test_data_filter_band,
                      noise_coeff=noise_coeff,
                      model_name=model_name,
                      save=True,
                      **best_params)


def optimize_and_test_real(model_name: str, optuna_objective: tp.Callable,
                           optuna_parameters_restructurer: tp.Optional[tp.Callable], dataset_path: str,
                           add_X_dimension: bool, test_function: tp.Callable, n_trials: int = 100):
    """
    Find the best parameters with optuna framework for the algorithm. Use them on the multi-person test dataset and
    save results

    Parameters
    ----------
    model_name : str
                Name of the model. Typically used by a test_function to write to appropriate model folder in the
                results directory results/test_results/model_name/YYYY-MM-DD/00-00-00
    optuna_objective : Callable
                Objective function which returns appropriate metric value
    optuna_parameters_restructurer : Callable or None
                A function which restructures a dictionary with model parameters. If not needed to restructure use None
    dataset_path : str
                Path to the multi-person real dataset
    add_X_dimension : bool
                Whether to add additional dimension to the input time-series. If False input time-series shape
                is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications
    test_function : Callable
                A function used to run a single test for a specific hyperparameters of a filter and calculate metrics
    n_trials : int
                Amount of trials for optimization
    """
    people_indices = [0, 11, 22, 30, 49]
    sampling_rate = 250
    dataset_type = 'multiperson_real_data'
    dataset_name = 'multiperson_real_data'
    metric_name = 'correlation'
    frequency = None
    filter_band = None
    noise_coeff = None

    per_person_dict = {}
    for person_idx in people_indices:
        np_train_ds, np_test_ds = get_multiperson_datasets(dataset_path=dataset_path,
                                                           person_idx=person_idx,
                                                           add_X_dimension=add_X_dimension,
                                                           input_size=5024,
                                                           output_size=4000)

        best_params = get_best_params(metric_name=metric_name,
                                      optuna_additional_params={
                                          'sampling_rate': sampling_rate,
                                          'dataset': np_train_ds,
                                          'dataset_type': dataset_type,
                                          'dataset_name': dataset_name
                                      },
                                      optuna_n_trials=n_trials,
                                      optuna_objective=optuna_objective,
                                      optuna_parameters_restructurer=optuna_parameters_restructurer)

        detailed_errors_dict, predictions = test_function(sampling_rate=sampling_rate,
                                                          dataset=np_test_ds,
                                                          dataset_type=dataset_type,
                                                          dataset_name=dataset_name,
                                                          frequency=frequency,
                                                          filter_band=filter_band,
                                                          noise_coeff=noise_coeff,
                                                          model_name=model_name,
                                                          save=False,
                                                          **best_params)

        for param_name, param_value in best_params.items():
            detailed_errors_dict[param_name] = param_value

        per_person_dict[person_idx] = {'detailed_errors_dict': detailed_errors_dict,
                                       'predictions': predictions}

    save_multipers_results(per_person_dict=per_person_dict,
                           dataset_type=dataset_type,
                           dataset_name=dataset_name,
                           sampling_rate=sampling_rate,
                           model_name=model_name)

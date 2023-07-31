import typing as tp
import numpy as np
import torch
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from tqdm import tqdm
from eeg_forecast.compatible_dataset import DartsCompatibleGeneratedDataset, DartsCompatibleSyntheticDataset
from general_utilities.metrics import calculate_batched_circstd, calculate_batched_correlation, calculate_batched_plv
from general_utilities.utils import calculate_statistics


def forecast_with_darts(model: TorchForecastingModel,
                        dataset: tp.Union[DartsCompatibleGeneratedDataset, DartsCompatibleSyntheticDataset],
                        amount_of_steps: int, num_loader_workers: int, batch_size: int):
    """
    Do per sample forecast of each time-series and collect predictions into arrays

    Parameters
    ----------
    model : TorchForecastingModel
                Forecasting model from a darts library
    dataset : DartsCompatibleGeneratedDataset or DartsCompatibleSyntheticDataset
                Darts compatible dataset which will be used for forecasting
    amount_of_steps : int
                Amount of steps for forecasting (forecasting horizon)
    num_loader_workers : int
                Amount of loader workers for prediction
    batch_size : int
                Batch size for prediction

    Returns
    -------
    forecasts : list
                List of arrays with predictions for every dataset time-series
    """
    test_target_ts, test_covariates = dataset.get_test_target_and_cov(model.input_chunk_length)
    ds_size = len(test_target_ts)
    forecasts = []
    for idx in tqdm(range(ds_size)):
        target_ts = test_target_ts[idx]
        covariates = test_covariates[idx]
        predictions = model.predict(amount_of_steps, series=target_ts, past_covariates=covariates,
                                    num_loader_workers=num_loader_workers, batch_size=batch_size)
        forecasts.append(np.stack([pred.values() for pred in predictions], axis=0))
    return forecasts


def remove_unnecessary_np_points(gt_ts_list: list, predict_ts_list: list, amount_of_steps: int,
                                 input_chunk_length: int):
    """
    Aligns ground truth and predictions and remove some ground truth and predictions which cannot be aligned.
    For example, some points in ground truth arrays come too early and there is not enough lags in models inputs to
    forecast them. On the other hand, some forecasted points are too far in the future and there is not enough ground
    truth to evaluate them

    Parameters
    ----------
    gt_ts_list : list
                List of arrays with ground truth for every time-series
    predict_ts_list : list
                List of arrays with predictions for every time-series
    amount_of_steps : int
                Amount of steps for forecasting (forecasting horizon)
    input_chunk_length : int
                Amount of lags which a model will require as an input to predict final point

    Returns
    -------
    gt : ndarray
                Array with aligned ground truth for every time-series
    preds : ndarray
                Array with aligned predictions for every time-series
    """
    assert amount_of_steps > 0
    gt = np.array([gt_ts.values() for gt_ts in gt_ts_list])
    preds = np.array(predict_ts_list)[:, :-amount_of_steps, -1, :]
    gt_start = input_chunk_length - 1 + amount_of_steps
    gt = gt[:, gt_start:, :]
    assert gt.shape == preds.shape
    return gt, preds


def get_test_metrics(gt_ts_list: list, predict_ts_list: list, amount_of_steps: int, input_chunk_length: int):
    """
    Aligns ground truth and predictions and remove some ground truth and predictions which cannot be aligned.
    For example, some points in ground truth arrays come too early and there is not enough lags in models inputs to
    forecast them. On the other hand, some forecasted points are too far in the future and there is not enough ground
    truth to evaluate them

    Parameters
    ----------
    gt_ts_list : list
                List of arrays with ground truth for every time-series
    predict_ts_list : list
                List of arrays with predictions for every time-series
    amount_of_steps : int
                Amount of steps for forecasting (forecasting horizon)
    input_chunk_length : int
                Amount of lags which a model will require as an input to predict final point

    Returns
    -------
    detailed_errors_dict : dict
                Dictionary which has lists of error for appropriate metrics
                ['correlation', 'plv', 'circstd', 'circstd_degrees']
    errors_dict : dict
                Dictionary which has aggregated error values for appropriate metrics
    """
    numpy_gt, numpy_predict = remove_unnecessary_np_points(gt_ts_list=gt_ts_list,
                                                           predict_ts_list=predict_ts_list,
                                                           amount_of_steps=amount_of_steps,
                                                           input_chunk_length=input_chunk_length)

    assert numpy_gt.shape == numpy_predict.shape, (numpy_gt.shape, numpy_predict.shape)

    torch_gt = torch.tensor(numpy_gt, dtype=torch.float32)
    torch_predict = torch.tensor(numpy_predict, dtype=torch.float32)

    complex_pred = torch_predict[:, :, 0] + 1j * torch_predict[:, :, 1]
    complex_gt = torch_gt[:, :, 0] + 1j * torch_gt[:, :, 1]

    envelope_pred = torch.abs(complex_pred)
    envelope_gt = torch.abs(complex_gt)

    correlations_per_batch = calculate_batched_correlation(envelope_pred, envelope_gt)
    plvs_per_batch = calculate_batched_plv(complex_pred, complex_gt)
    circstd_per_batch = calculate_batched_circstd(complex_pred, complex_gt)
    circstd_per_batch_degrees = circstd_per_batch * 180. / np.pi

    results_corr = calculate_statistics(correlations_per_batch, ['min', 'max', 'mean', 'median'])
    results_plvs = calculate_statistics(plvs_per_batch, ['min', 'max', 'mean', 'median'])
    results_circstd = calculate_statistics(circstd_per_batch, ['min', 'max', 'mean', 'median'])
    results_circstd_degrees = calculate_statistics(circstd_per_batch_degrees, ['min', 'max', 'mean', 'median'])

    errors_dict = {}

    for metric_name, metric_value in results_corr.items():
        errors_dict[metric_name + '_correlation'] = metric_value

    for metric_name, metric_value in results_plvs.items():
        errors_dict[metric_name + '_plv'] = metric_value

    for metric_name, metric_value in results_circstd.items():
        errors_dict[metric_name + '_circstd'] = metric_value

    for metric_name, metric_value in results_circstd_degrees.items():
        errors_dict[metric_name + '_circstd_degrees'] = metric_value

    detailed_errors_dict = {
        'correlation': correlations_per_batch.cpu().numpy().tolist(),
        'plv': plvs_per_batch.cpu().numpy().tolist(),
        'circstd': circstd_per_batch.cpu().numpy().tolist(),
        'circstd_degrees': circstd_per_batch_degrees.cpu().numpy().tolist()
    }

    return detailed_errors_dict, errors_dict


def save_predictions_gt(filename: str, gt_ts_list: list, predict_ts_list: list, amount_of_steps: int,
                        input_chunk_length: int):
    """
    Saved ground truth and predictions in npz file

    Parameters
    ----------
    filename : str
                Path to the file to write predictions and ground truth to
    gt_ts_list : list
                List of arrays with ground truth for every time-series
    predict_ts_list : list
                List of arrays with predictions for every time-series
    amount_of_steps : int
                Amount of steps for forecasting (forecasting horizon)
    input_chunk_length : int
                Amount of lags which a model will require as an input to predict final point
    """
    numpy_gt, numpy_predict = remove_unnecessary_np_points(gt_ts_list=gt_ts_list,
                                                           predict_ts_list=predict_ts_list,
                                                           amount_of_steps=amount_of_steps,
                                                           input_chunk_length=input_chunk_length)

    assert numpy_gt.shape == numpy_predict.shape, (numpy_gt.shape, numpy_predict.shape)

    complex_pred = numpy_predict[:, :, 0] + 1j * numpy_predict[:, :, 1]
    complex_gt = numpy_gt[:, :, 0] + 1j * numpy_gt[:, :, 1]

    envelope_pred = np.abs(complex_pred)
    envelope_gt = np.abs(complex_gt)

    predictions_dict = {'complex_pred': complex_pred,
                        'complex_gt': complex_gt,
                        'envelope_pred': envelope_pred,
                        'envelope_gt': envelope_gt}

    np.savez(filename, **predictions_dict)

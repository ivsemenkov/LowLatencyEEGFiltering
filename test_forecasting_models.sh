#!/bin/bash
export PYTHONPATH="$(pwd)"

# TCNModel Synthetic
python eeg_forecast/test_forecasting_model.py --dataset_name sines_white --path results/outputs/TCNModel/2023-07-25/15-34-36
python eeg_forecast/test_forecasting_model.py --dataset_name sines_pink --path results/outputs/TCNModel/2023-07-26/20-32-37
python eeg_forecast/test_forecasting_model.py --dataset_name filtered_pink --path results/outputs/TCNModel/2023-07-28/03-50-37
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_white --path results/outputs/TCNModel/2023-07-28/16-06-07
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_pink --path results/outputs/TCNModel/2023-07-29/02-29-24


# DLinearModel Synthetic
python eeg_forecast/test_forecasting_model.py --dataset_name sines_white --path results/outputs/DLinearModel/2023-07-26/05-01-00
python eeg_forecast/test_forecasting_model.py --dataset_name sines_pink --path results/outputs/DLinearModel/2023-07-27/09-14-41
python eeg_forecast/test_forecasting_model.py --dataset_name filtered_pink --path results/outputs/DLinearModel/2023-07-28/06-00-14
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_white --path results/outputs/DLinearModel/2023-07-28/17-58-46
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_pink --path results/outputs/DLinearModel/2023-07-29/04-50-45


# TransformerModel Synthetic
python eeg_forecast/test_forecasting_model.py --dataset_name sines_white --path results/outputs/TransformerModel/2023-07-26/10-16-44
python eeg_forecast/test_forecasting_model.py --dataset_name sines_pink --path results/outputs/TransformerModel/2023-07-27/16-43-04
python eeg_forecast/test_forecasting_model.py --dataset_name filtered_pink --path results/outputs/TransformerModel/2023-07-28/08-33-50
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_white --path results/outputs/TransformerModel/2023-07-28/19-38-56
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_pink --path results/outputs/TransformerModel/2023-07-29/06-35-54


# NHiTSModel Synthetic
python eeg_forecast/test_forecasting_model.py --dataset_name sines_white --path results/outputs/NHiTSModel/2023-07-26/13-16-47
python eeg_forecast/test_forecasting_model.py --dataset_name sines_pink --path results/outputs/NHiTSModel/2023-07-27/18-55-25
python eeg_forecast/test_forecasting_model.py --dataset_name filtered_pink --path results/outputs/NHiTSModel/2023-07-28/10-58-24
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_white --path results/outputs/NHiTSModel/2023-07-28/23-33-11
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_pink --path results/outputs/NHiTSModel/2023-07-29/08-29-06


# NLinearModel Synthetic
python eeg_forecast/test_forecasting_model.py --dataset_name sines_white --path results/outputs/NLinearModel/2023-07-26/14-41-43
python eeg_forecast/test_forecasting_model.py --dataset_name sines_pink --path results/outputs/NLinearModel/2023-07-27/20-26-32
python eeg_forecast/test_forecasting_model.py --dataset_name filtered_pink --path results/outputs/NLinearModel/2023-07-28/12-58-01
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_white --path results/outputs/NLinearModel/2023-07-29/00-59-48
python eeg_forecast/test_forecasting_model.py --dataset_name state_space_pink --path results/outputs/NLinearModel/2023-07-29/10-06-39

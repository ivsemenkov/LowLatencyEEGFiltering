#!/bin/bash
export PYTHONPATH="$(pwd)"
DATASET_PATH="$(pwd)/Datasets/real_data_250Hz.npz"

# TCNFilteringModel Synthetic
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name sines_white --path results/outputs/TCNFilteringModel/2023-07-24/20-13-04
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name sines_pink --path results/outputs/TCNFilteringModel/2023-07-25/07-08-26
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name filtered_pink --path results/outputs/TCNFilteringModel/2023-07-25/22-13-25
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name state_space_white --path results/outputs/TCNFilteringModel/2023-07-26/09-14-09
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name state_space_pink --path results/outputs/TCNFilteringModel/2023-07-26/20-19-12


# ConvTasNetFilteringModel Synthetic
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name sines_white --path results/outputs/ConvTasNetFilteringModel/2023-07-25/02-21-48
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name sines_pink --path results/outputs/ConvTasNetFilteringModel/2023-07-25/10-54-38
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name filtered_pink --path results/outputs/ConvTasNetFilteringModel/2023-07-26/00-54-00
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name state_space_white --path results/outputs/ConvTasNetFilteringModel/2023-07-26/11-12-29
python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name state_space_pink --path results/outputs/ConvTasNetFilteringModel/2023-07-26/22-13-30


# TCNFilteringModel Real
python eeg_nn_filtering/test_filtering_model_real.py --dataset_path "${DATASET_PATH}" --path results/outputs/TCNFilteringModel/2023-07-27/06-51-01


# ConvTasNetFilteringModel Real
python eeg_nn_filtering/test_filtering_model_real.py --dataset_path "${DATASET_PATH}" --path results/outputs/ConvTasNetFilteringModel/2023-07-27/10-14-40

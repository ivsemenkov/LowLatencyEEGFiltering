#!/bin/bash
export PYTHONPATH="$(pwd)"
ROOT="$(pwd)/config_files/forecasting_config_files"
DATASET_NAMES=("sines_white" "sines_pink" "filtered_pink" "state_space_white" "state_space_pink")
MODEL_NAMES=("TCN" "DLinear" "Transformer" "NHiTS" "NLinear")
for dataset_name in "${DATASET_NAMES[@]}" ; do
    for model_name in "${MODEL_NAMES[@]}" ; do
        CONFIG_PATH="${ROOT}/${dataset_name}"
        CONFIG_NAME="config_${model_name}.yaml"
        echo "${CONFIG_PATH}"
        echo "${CONFIG_NAME}"
        python eeg_forecast/train_forecasting_model.py --config-path "${CONFIG_PATH}" --config-name "${CONFIG_NAME}"
    done
done

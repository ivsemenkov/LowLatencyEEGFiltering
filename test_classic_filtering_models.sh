#!/bin/bash
export PYTHONPATH="$(pwd)"

# cFIR Synthetic
python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name sines_white --dataset_path Datasets/sines_white_10Hz_250Hz.npz
python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name sines_pink --dataset_path Datasets/sines_pink_10Hz_250Hz.npz
python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name state_space_white --dataset_path Datasets/state_space_white_10Hz_250Hz.npz
python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name state_space_pink --dataset_path Datasets/state_space_pink_10Hz_250Hz.npz
python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name filtered_pink --dataset_path Datasets/pink_filt_8_12_250Hz.npz


# cFIR Real
python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name multiperson_real_data --dataset_path Datasets/real_data_250Hz.npz


# Kalman filter Synthetic
python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name sines_white --dataset_path Datasets/sines_white_10Hz_250Hz.npz
python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name sines_pink --dataset_path Datasets/sines_pink_10Hz_250Hz.npz
python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name state_space_white --dataset_path Datasets/state_space_white_10Hz_250Hz.npz
python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name state_space_pink --dataset_path Datasets/state_space_pink_10Hz_250Hz.npz
python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name filtered_pink --dataset_path Datasets/pink_filt_8_12_250Hz.npz


# Kalman filter Real
python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name multiperson_real_data --dataset_path Datasets/real_data_250Hz.npz

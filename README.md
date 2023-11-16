# Low latency EEG filtering

This repository provides all code used for the paper "Real-time low latency estimation of brain rhythms with deep 
neural networks"

## Configs

All configs for neural networks are provided in config_files directory. This directory is split into the following subdirectories:
* filtering_config_files - config files used for EEG filtering with TCN or Conv-TasNet models
* forecasting_config_files - config files used for synthetic EEG forecasting

## Datasets

Datasets used in the experiments are provided here https://doi.org/10.5281/zenodo.8135837

## Synthetic EEG forecasting

The synthetic EEG forecasting experiments refer to the experiments where the true synthetic ground truth analytic 
signal is being artificially delayed by 100 ms relative to the noisy observations. Each model then receives a delayed 
ground truth signal coupled with real-time noisy observations. Each neural network then forecasts the ground truth time
series 100 ms forward to compensate the introduced delay.

Code specific to these experiments is provided in the `eeg_forecast` directory.

Config files for these experiments are located in the `config_files/forecasting_config_files` directory.
Path to each config file is of the following structure 
`config_files/forecasting_config_files/{dataset_name}/config_{model_name}.yaml`. 

Available `dataset_name`:
* sines_white - Sines white experiment: sine wave at 10 Hz with white noise
* sines_pink - Sines pink experiment: sine wave at 10 Hz with pink noise
* filtered_pink - Filtered pink experiment: one sample of pink noise is filtered in the band 8-12 Hz and taken as 
ground truth. Second sample of pink noise is added as noise
* state_space_white - State-space white experiment: ground truth obtained with a state-space model at 10 Hz central 
frequency with white noise
* state_space_pink - State-space pink experiment: ground truth obtained with a state-space model at 10 Hz central 
frequency with pink noise

Available `model_name`:
* DLinear
* NLinear
* Temporal Convolutional Network (TCN)
* NHiTS
* Transformer

### Training 

To run the training stage:

`export PYTHONPATH="$(pwd)"`

`python eeg_forecast/train_forecasting_model.py --config-path {config_path} --config-name {config_name}`

If a path to your config is `config_files/forecasting_config_files/{dataset_name}/config_{model_name}.yaml`, then
`config_path`=`config_files/forecasting_config_files/{dataset_name}` and 
`config_name`=`config_{model_name}.yaml`

To train all models on all available datasets from default config paths you can also use the 
`train_forecasting_models.sh` script

### Testing

To run the testing stage:

`export PYTHONPATH="$(pwd)"`

`python eeg_forecast/test_forecasting_model.py --dataset_name {dataset_name} --path {trained_model_path}`

`dataset_name` is a name of one of the available synthetic datasets to test already trained model on.

`trained_model_path` is a path to the results of the training stage. Typically, it will look like 
`results/outputs/{model_name}/YYYY-MM-DD/00-00-00`

Check out the file `eeg_forecast/test_forecasting_model.py` to see some additional arguments available

You can use file `general_utilities/get_test_paths.py --models {models_list} --dates {dates} --model_type forecasting`
to generate a list of commands for various models `models_list` and experiments done on different dates
`dates`. It will accumulate all forecasting experiments for every model in `models_list` for dates in `dates` list
and print a list of commands which could be combined into a bash script like `test_forecasting_models.sh`


## Synthetic and real EEG filtration

The synthetic and real EEG filtration experiments refer to the experiments where the models get only real-time 
observations as an input and must output complex analytic signal. These experiments were done with classic filtering
models like cFIR and Kalman filter proven to provide good quality results; and with modern neural networks: 
Temporal Convolutional Network (TCN) which shown overall strongest results in synthetic EEG forecasting experiments and
Conv-TasNet which is a significantly larger State-of-the-Art model based on the TCN blocks

### Classic models

Code specific to these experiments for classic models is provided in the `eeg_classic_filtering` directory

To optimize hyperparameters of a cFIR filter and test it on a specific `dataset_name` run the following commands:

`export PYTHONPATH="$(pwd)"`

`python eeg_classic_filtering/cfir_baseline_experiment.py --dataset_name {dataset_name} --dataset_path {dataset_path}`

For Kalman filter use:

`export PYTHONPATH="$(pwd)"`

`python eeg_classic_filtering/kalman_baseline_experiment.py --dataset_name {dataset_name} --dataset_path {dataset_path}`

`dataset_name` is a name of the dataset which will be generated to test a model on (for `multiperson_real_data`
testing will be done on the dataset provided in the `dataset_path` on a test split)

`dataset_path` is a path to the dataset on which models hyperparameters should be optimized (training data)

Available `dataset_name` and their default expected `dataset_path`:
* sines_white - Sines white experiment: sine wave at 10 Hz with white noise; `Datasets/sines_white_10Hz_250Hz.npz`
* sines_pink - Sines pink experiment: sine wave at 10 Hz with pink noise; `Datasets/sines_pink_10Hz_250Hz.npz`
* filtered_pink - Filtered pink experiment: one sample of pink noise is filtered in the band 8-12 Hz and taken as 
ground truth. Second sample of pink noise is added as noise; `Datasets/pink_filt_8_12_250Hz.npz`
* state_space_white - State-space white experiment: ground truth obtained with a state-space model at 10 Hz central 
frequency with white noise; `Datasets/state_space_white_10Hz_250Hz.npz`
* state_space_pink - State-space pink experiment: ground truth obtained with a state-space model at 10 Hz central 
frequency with pink noise; `Datasets/state_space_pink_10Hz_250Hz.npz`
* multiperson_real_data - Real EEG experiment: has data from 25 people involved in the P4 alpha neurofeedback training.
5 of them are used for model testing; `Datasets/real_data_250Hz.npz`

Available models:
* cFIR
* Kalman filter

To run all models on all datasets you can also use `test_classic_filtering_models.sh`

### Neural networks 

Code specific to these experiments is provided in the `eeg_nn_filtering` directory.

Config files for these experiments are located in the `config_files/filtering_config_files` directory.
Path to each config file is of the following structure 
`config_files/filtering_config_files/{dataset_name}/config_{model_name}.yaml`. 

Available `dataset_name`:
* sines_white - Sines white experiment: sine wave at 10 Hz with white noise
* sines_pink - Sines pink experiment: sine wave at 10 Hz with pink noise
* filtered_pink - Filtered pink experiment: one sample of pink noise is filtered in the band 8-12 Hz and taken as 
ground truth. Second sample of pink noise is added as noise
* state_space_white - State-space white experiment: ground truth obtained with a state-space model at 10 Hz central 
frequency with white noise
* state_space_pink - State-space pink experiment: ground truth obtained with a state-space model at 10 Hz central 
frequency with pink noise
* multiperson_real_data - Real EEG experiment: has data from 25 people involved in the P4 alpha neurofeedback training.
20 are used to train the model; 5 out-of-sample are used for model fine-tuning and testing

Available models:
* Temporal Convolutional Network (TCN)
* Conv-TasNet (ConvTasNet)

#### Training 

To run the training stage:

`export PYTHONPATH="$(pwd)"`

`python eeg_nn_filtering/train_filtering_model.py --config-path {config_path} --config-name {config_name}`

If a path to your config is `config_files/forecasting_config_files/{dataset_name}/config_{model_name}.yaml`, then
`config_path`=`config_files/forecasting_config_files/{dataset_name}` and 
`config_name`=`config_{model_name}.yaml`

To train all models on all available datasets from default config paths you can also use the 
`train_filtering_nn_models.sh` script

#### Testing

To run the testing stage for synthetic data use:

`export PYTHONPATH="$(pwd)"`

`python eeg_nn_filtering/test_filtering_model_synthetic.py --dataset_name {dataset_name} --path {trained_model_path}`

`dataset_name` is a name of the synthetic dataset which will be generated to test already trained model on

`trained_model_path` is a path to the results of the training stage. Typically, it will look like 
`results/outputs/{model_name}/YYYY-MM-DD/00-00-00`

To run the testing stage for real data use:

`export PYTHONPATH="$(pwd)"`

`python eeg_nn_filtering/test_filtering_model_real.py --dataset_path {dataset_path} --path {trained_model_path}`

`dataset_path` is a path to the real data to test already trained model on. Typically, it will be 
`Datasets/real_data_250Hz.npz`

`trained_model_path` is a path to the results of the training stage. Typically, it will look like 
`results/outputs/{model_name}/YYYY-MM-DD/00-00-00`

Check out files `eeg_nn_filtering/test_filtering_model_synthetic.py` and 
`eeg_nn_filtering/test_filtering_model_real.py` to see some additional arguments available

You can use file `general_utilities/get_test_paths.py --models {models_list} --dates {dates} --model_type filtering`
to generate a list of commands for various models `models_list` and experiments done on different dates
`dates`. It will accumulate all forecasting experiments for every model in `models_list` for dates in `dates` list
and print a list of commands which could be combined into a bash script like `test_filtering_nn_models.sh`

## Visualizations

To avoid significantly longer computation of metrics and statistics for the graph visualization process is split
into computation of supplementary csv files and plotting graphs. Visualizations generated like in the original paper.

Code specific to visualizations is provided in the `visualization` directory.

Initially you must prepare separate txt files for forecasting and filtering experiments with paths to 
json files with metrics after testing stage. Running testing stage for forecasting, classic models filtering and 
neural network filtering will generate json files with detailed test metrics at paths like 
`results/test_results/{model_name}/YYYY-MM-DD/00-00-00/test_metrics_{idx}.json`. You should choose which 
experiments to include into graphs by putting all the paths to use into a txt file in 
`config_files/visualization_config_files`. Classic filtering and neural network filtering experiments should 
be present in the same file (example: `config_files/visualization_config_files/filtering_filepaths.txt`).
Forecasting experiments should be in a separate file (example: 
`config_files/visualization_config_files/forecasting_filepaths.txt`). Do not put multiple paths to test metrics for 
the same experiment setup. For example, if you tested 2 times TCN on sines_white dataset with the same noise 
coefficient choose one to put in the file. Script will at some point drop information about paths and rely on 
information about experiment setup only which might lead to merging multiple runs on the same setup. Note, in 
Filtered pink and State-space pink experiments there will be multiple test_metrics files generated for different noise
coefficients. Here, due to the different noise coefficients the setups are different, so add all of them. 
Also, check out the example files `config_files/visualization_config_files/filtering_filepaths.txt` and
`config_files/visualization_config_files/forecasting_filepaths.txt`. 

To create supplementary csv files run:

`export PYTHONPATH="$(pwd)"`

`python visualization/create_plotting_data.py --path {txt_path} --model_type {model_type}`

`txt_path` is a path to the txt file with chosen experiments.

`model_type` is `filtering` if `txt_path` is provided for filtering experiments; or`forecasting` if `txt_path` is for 
forecasting experiments

To plot and save graphs afterwards run:

`export PYTHONPATH="$(pwd)"`

`python visualization/plot_graphs.py --path {supplementary_dir} --model_type {model_type}`

`supplementary_dir` is a path to the directory with the created supplementary csv files. Typically, it will be
`results/visualization_csv_files`.

`model_type` is `filtering` or `forecasting` depending on which graphs do you want to plot.

Graphs will be stored in `results/paper_graphs/{model_type}`.

Note, this script will save full graphs and also each of their subgraph separately. This is useful as such subgraphs 
can be inserted separately into LaTeX file and can be assigned different labels to make navigation easier

## Additional information

* Some provided scripts **do not** utilize parallelism. This is done to preserve reproducibility better. 
However, you might want to fix this if reproducibility is not crucial to your runs.
* It is recommended to run scripts from the project root to avoid misplacement of files created during train and test.
* To check out the HFIR algorithm which was used to generate ground truth for real data and training ground truth for
synthetic data see file `general_utilities/gt_extraction.py`.

## Paper

If you use our data or code please cite: 

https://iopscience.iop.org/article/10.1088/1741-2552/acf7f3

@article{Semenkov_2023,
doi = {10.1088/1741-2552/acf7f3},
url = {https://dx.doi.org/10.1088/1741-2552/acf7f3},
year = {2023},
month = {sep},
publisher = {IOP Publishing},
volume = {20},
number = {5},
pages = {056008},
author = {Ilia Semenkov and Nikita Fedosov and Ilya Makarov and Alexei Ossadtchi},
title = {Real-time low latency estimation of brain rhythms with deep neural networks},
journal = {Journal of Neural Engineering}
}


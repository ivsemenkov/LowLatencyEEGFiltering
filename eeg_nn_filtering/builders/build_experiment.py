import typing as tp
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, open_dict
from .build_dataset import build_dataloader
from eeg_nn_filtering.networks import ConvTasNetFilteringModel, EEGFilteringSystem, TCNFilteringModel


def build_eeg_model(model_name: str, model_params: dict) -> nn.Module:
    """
    Returns initialized EEG filtering neural network

    Parameters
    ----------
    model_name : str
                Name of the model
    model_params : dict
                Model-specific parameters

    Returns
    -------
    model : nn.Module
                Initialized model
    """
    if model_name == 'TCNFilteringModel':
        model = TCNFilteringModel(**model_params)
    elif model_name == 'ConvTasNetFilteringModel':
        model = ConvTasNetFilteringModel(**model_params)
    else:
        raise NotImplementedError
    return model


def build_experiment(config: tp.Union[dict, DictConfig],
                     dataset_path: tp.Optional[str],
                     split_ds_into_chunks: bool):
    """
    Get Pytorch Lightning model and dataloaders for training, validation and testing

    Parameters
    ----------
    config : dict or DictConfig
                Configuration file for an experiment
    dataset_path : str
                Path to the dataset
    split_ds_into_chunks : bool
                If True datasets will return specific chunks of length equal to the models input size instead of
                larger time-series which are supposed to be filtered one sample at a time in correct order. This
                is useful for training as the whole data is available from the beginning, and it helps to use less
                GPU space by holding fewer data in each batch at the same time.

    Returns
    -------
    system : LightningModule
                Pytorch Lightning model
    train_loader : DataLoader
                DataLoader with training data
    val_loader : DataLoader
                DataLoader with validation data
    test_loader : DataLoader
                DataLoader with testing data
    """
    dataset_common_params = config['dataset_common_params']
    dataset_type = config['dataset_type']
    train_dataset_params = config['train_dataset_specific_params']
    val_dataset_params = config['val_dataset_specific_params']
    test_dataset_params = config['test_dataset_specific_params']
    norm_by_train_std = config['norm_by_train_std']
    model_params = config['model_params']

    if split_ds_into_chunks:
        input_chunk_length = model_params['input_chunk_length']
        output_chunk_length = model_params['output_chunk_length']
        with open_dict(train_dataset_params):
            train_dataset_params['split_ds_into_chunks'] = True
            train_dataset_params['input_chunk_length'] = input_chunk_length
            train_dataset_params['output_chunk_length'] = output_chunk_length
        with open_dict(val_dataset_params):
            val_dataset_params['split_ds_into_chunks'] = True
            val_dataset_params['input_chunk_length'] = input_chunk_length
            val_dataset_params['output_chunk_length'] = output_chunk_length
        with open_dict(test_dataset_params):
            test_dataset_params['split_ds_into_chunks'] = False
            test_dataset_params['input_chunk_length'] = None
            test_dataset_params['output_chunk_length'] = None

    eeg_model = build_eeg_model(model_name=config['model'],
                                model_params=model_params)

    train_loader, train_input_std = build_dataloader(common_params_dict=dataset_common_params,
                                                     dataset_path=dataset_path,
                                                     dataset_type=dataset_type,
                                                     dataset_params=train_dataset_params,
                                                     mode='train',
                                                     norm_by_train_std=norm_by_train_std,
                                                     train_input_std=None,
                                                     sampler_params=config['train_sampler_params'],
                                                     dataloader_params=config['dataloader_params'],
                                                     shuffle=True,
                                                     tag='Train')

    val_loader, _ = build_dataloader(common_params_dict=dataset_common_params,
                                     dataset_path=dataset_path,
                                     dataset_type=dataset_type,
                                     dataset_params=val_dataset_params,
                                     mode='val',
                                     norm_by_train_std=norm_by_train_std,
                                     train_input_std=train_input_std,
                                     sampler_params=config['val_sampler_params'],
                                     dataloader_params=config['dataloader_params'],
                                     shuffle=False,
                                     tag='Validation')

    test_loader, _ = build_dataloader(common_params_dict=dataset_common_params,
                                      dataset_path=dataset_path,
                                      dataset_type=dataset_type,
                                      dataset_params=test_dataset_params,
                                      mode='test',
                                      norm_by_train_std=norm_by_train_std,
                                      train_input_std=train_input_std,
                                      sampler_params=None,
                                      dataloader_params=config['dataloader_params'],
                                      shuffle=False,
                                      tag='Test')

    optimizer = optim.AdamW(eeg_model.parameters(), **config['opt_params'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['sch_params'])

    system = EEGFilteringSystem(eeg_filtering_model=eeg_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                criterion=nn.MSELoss(),
                                **config['system_params'])

    return system, train_loader, val_loader, test_loader

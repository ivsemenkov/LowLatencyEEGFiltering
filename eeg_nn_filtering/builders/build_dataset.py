import typing as tp
import torch
from torch.utils.data import DataLoader, RandomSampler
from eeg_nn_filtering.data import GeneratedSyntheticDataset, PreSplitDataset, RealEEGMultiPersonDataset
from general_utilities.utils import fill_specific_params_dict


def build_core_dataset(dataset_path: str,
                       dataset_type: str,
                       dataset_params: dict):
    """
    Get an initialized dataset

    Parameters
    ----------
    dataset_path : str
            Path to the dataset
    dataset_type : str
            Type of the dataset. 'generated' to generate a new synthetic dataset. 'pre_split' to load and initialize a
            synthetic dataset generated and saved already. 'multiperson_real_data' for real dataset
    dataset_params : dict
            Dictionary with dataset-specific parameters

    Returns
    -------
    ds : GeneratedSyntheticDataset or PreSplitDataset or RealEEGMultiPersonDataset
            An initialized dataset
    """
    if dataset_type == 'generated':
        ds = GeneratedSyntheticDataset(**dataset_params)
    elif dataset_type == 'multiperson_real_data':
        ds = RealEEGMultiPersonDataset(dataset_path=dataset_path,
                                       **dataset_params)
    elif dataset_type == 'pre_split':
        ds = PreSplitDataset(dataset_path=dataset_path,
                             **dataset_params)
    else:
        raise NotImplementedError
    return ds


def build_dataset(dataset_path: str,
                  dataset_type: str,
                  dataset_params: dict,
                  mode: str,
                  norm_by_train_std: bool,
                  train_input_std: tp.Optional[float]):
    """
    Get a finalized dataset

    Parameters
    ----------
    dataset_path : str
            Path to the dataset
    dataset_type : str
            Type of the dataset. 'generated' to generate a new synthetic dataset. 'pre_split' to load and initialize a
            synthetic dataset generated and saved already. 'multiperson_real_data' for real dataset
    dataset_params : dict
            Dictionary with dataset-specific parameters
    mode : str
            How the dataset is used: 'train', 'val' or 'test'
    norm_by_train_std : bool
            If True will normalize inputs by their standard deviation (if mode == 'train') or by provided
            train_input_std ('val' and 'test')
    train_input_std : float or None
            If norm_by_train_std is True and mode is different from 'train' this number will be used as a normalization
            factor for inputs. Suggested to provide here standard deviation of the train dataset as it is assumed
            that we cannot get standard deviation of the whole time-series during validation and test in real-time
            applications

    Returns
    -------
    ds : GeneratedSyntheticDataset or PreSplitDataset or RealEEGMultiPersonDataset
            An finalized dataset with or without normalization
    train_input_std : float or None
            Standard deviation of the dataset (if mode == 'train') or provided train_input_std otherwise
    """
    assert mode in ['train', 'val', 'test'], 'Modes available: train, val, test'
    ds = build_core_dataset(dataset_path=dataset_path,
                            dataset_type=dataset_type,
                            dataset_params=dataset_params)

    if norm_by_train_std:
        if mode == 'train':
            train_input_std = ds.get_ds_std()
        else:
            ds.set_ds_std(train_input_std)
        ds.scale_inputs_by_std()

    return ds, train_input_std


def build_dataloader(common_params_dict: dict,
                     dataset_path: str,
                     dataset_type: str,
                     dataset_params: dict,
                     mode: str,
                     norm_by_train_std: bool,
                     train_input_std: tp.Optional[float],
                     sampler_params: tp.Optional[dict],
                     dataloader_params: dict,
                     shuffle: bool,
                     tag: str):
    """
    Get a finalized dataset

    Parameters
    ----------
    common_params_dict : dict
            Dictionary with dataset-specific parameters common for train, validation and testing
    dataset_path : str
            Path to the dataset
    dataset_type : str
            Type of the dataset. 'generated' to generate a new synthetic dataset. 'pre_split' to load and initialize a
            synthetic dataset generated and saved already. 'multiperson_real_data' for real dataset
    dataset_params : dict
            Dictionary with dataset-specific parameters for this mode
    mode : str
            How the dataset is used: 'train', 'val' or 'test'
    norm_by_train_std : bool
            If True will normalize inputs by their standard deviation (if mode == 'train') or by provided
            train_input_std ('val' and 'test')
    train_input_std : float or None
            If norm_by_train_std is True and mode is different from 'train' this number will be used as a normalization
            factor for inputs. Suggested to provide here standard deviation of the train dataset as it is assumed
            that we cannot get standard deviation of the whole time-series during validation and test in real-time
            applications
    sampler_params : dict or None
            Dictionary with parameters for a sampler. If None, then sampler is not used
    dataloader_params : dict
            Dictionary with parameters for DataLoader
    shuffle : bool
            If True uses shuffle in DataLoader if sampler is not used
    tag : str
            Tag for this dataset for printing info if needed

    Returns
    -------
    full_loader : DataLoader
            An finalized DataLoader with or without normalization
    input_std : float
            Standard deviation of the dataset (if mode == 'train') or provided train_input_std otherwise
    """
    if dataset_params is None:
        return None, None
    else:
        fill_specific_params_dict(common_params_dict=common_params_dict,
                                  specific_params_dict=dataset_params)

        dataset, input_std = build_dataset(dataset_path=dataset_path,
                                           dataset_type=dataset_type,
                                           dataset_params=dataset_params,
                                           mode=mode,
                                           norm_by_train_std=norm_by_train_std,
                                           train_input_std=train_input_std)

        if sampler_params is not None:
            generator = torch.Generator()
            # Here PyTorch team recommends to use a large seed.
            # We use 2147483647 which is provided as an example in PyTorch docs.
            generator.manual_seed(2147483647)
            sampler = RandomSampler(data_source=dataset, generator=generator, **sampler_params)
            full_loader = DataLoader(dataset,
                                     sampler=sampler,
                                     **dataloader_params)
        else:
            full_loader = DataLoader(dataset,
                                     shuffle=shuffle,
                                     **dataloader_params)

        assert len(full_loader) > 0, f'{tag} loader has size {len(full_loader)}, try adjusting batch_size'
    return full_loader, input_std

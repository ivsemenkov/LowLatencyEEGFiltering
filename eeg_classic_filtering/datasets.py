import math
import numpy as np
import typing as tp
from eeg_nn_filtering.data import PreSplitDataset, RealEEGMultiPersonDataset, GeneratedSyntheticDataset


class NPWrapperDataset:
    def __init__(self, torch_dataset: tp.Union[PreSplitDataset, RealEEGMultiPersonDataset, GeneratedSyntheticDataset],
                 add_X_dimension: bool):
        """
        Takes a PyTorch dataset class and provides its observations (time-series) as numpy arrays instead of PyTorch
        tensors

        Parameters
        ----------
        torch_dataset : PreSplitDataset or RealEEGMultiPersonDataset
                    An instance of PyTorch dataset for transformation
        add_X_dimension : bool
                    Whether to add additional dimension to the input time-series. If False input time-series shape
                    is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications
        """
        self.torch_dataset = torch_dataset
        self.output_size = torch_dataset.output_size
        self.add_X_dimension = add_X_dimension

    def get_size(self):
        """
        Get amount of time-series (observations) in a dataset

        Returns
        -------
        size : int
                    Amount of observations
        """
        return len(self.torch_dataset)

    def get_datapoint(self, idx: int):
        """
        Get single time-series (observation) and its ground truth analytic signal

        Parameters
        ----------
        idx : int
                    Index of a required datapoint

        Returns
        -------
        input_data : ndarray
                    Time-series of unfiltered, noisy inputs
        gt_states : ndarray
                    Time-series of ground truth analytic signal
        """
        X, y = self.torch_dataset[idx]
        assert X.size() == (self.torch_dataset.input_size, 1)
        assert y.size() == (self.torch_dataset.output_size, 2)
        if not self.add_X_dimension:
            X = X[:, 0]
        y = y[:, 0] + 1.j * y[:, 1]
        return np.array(X), np.array(y)


def get_multiperson_datasets(dataset_path: str, person_idx: int, add_X_dimension: bool,
                             input_size: int = 5000, output_size: int = 5000):
    """
    Returns training and testing splits of multi-person real EEG dataset splits for person_idx. It will be a dataset
    for a single person where initial time steps are dedicated for training and later steps for testing. This imitates
    real-time framework where we cannot tune models on data which we do not yet have. Parameters coincide
    with ones for PyTorch experiments.

    Parameters
    ----------
    dataset_path : str
                Path to the multi-person real EEG dataset
    person_idx : int
                Integer id of a person for which the dataset is created
    add_X_dimension : bool
                Whether to add additional dimension to the input time-series. If False input time-series shape
                is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications
    input_size : int
                How long (in samples) should be each time-series of EEG data (there will be multiple of them).
                Each such time-series is processed separately one sample at a time to imitate real-time framework
    output_size : int
                How long (in samples) should be each time-series of HFIR filtered ground truth EEG data
                (there will be multiple of them). output_size should be <= input_size. If output_size < input_size then
                a ground truth series will consist of filtered states for output_size latest samples from input_size.
                For example, if input_size == 5 and output_size == 3 then each ground truth series will have filtered
                ground truth for indices 2, 3, 4 of input series. That might be useful if algorithm cannot filter every
                single sample (algorithm might need 999 more inputs as it uses 1000 lagged samples to filter the
                most fresh one)
    Returns
    -------
    np_train_ds : NPWrapperDataset
                Numpy version of RealEEGMultiPersonDataset for training
    np_test_ds : NPWrapperDataset
                Numpy version of RealEEGMultiPersonDataset for testing
    """

    train_dataset_params = {
        'dataset_path': dataset_path,
        'first_observation_idx': 0,
        'amount_of_observations': 45,
        'input_size': input_size,
        'output_size': output_size,
        'gap': 3250,
        'ids_to_use': [person_idx],
        'maximal_per_person_size': 90,
        'shuffle_dataset': False,
        'split_ds_into_chunks': False,
        'input_chunk_length': None,
        'output_chunk_length': None
    }

    train_ds = RealEEGMultiPersonDataset(**train_dataset_params)
    train_input_std = train_ds.get_ds_std()
    train_ds.scale_inputs_by_std()

    np_train_ds = NPWrapperDataset(torch_dataset=train_ds, add_X_dimension=add_X_dimension)

    test_dataset_params = {
        'dataset_path': dataset_path,
        'first_observation_idx': 54,
        'amount_of_observations': 36,
        'input_size': input_size,
        'output_size': output_size,
        'gap': 3250,
        'ids_to_use': [person_idx],
        'maximal_per_person_size': 90,
        'shuffle_dataset': False,
        'split_ds_into_chunks': False,
        'input_chunk_length': None,
        'output_chunk_length': None
    }

    test_ds = RealEEGMultiPersonDataset(**test_dataset_params)
    test_ds.set_ds_std(train_input_std)
    test_ds.scale_inputs_by_std()

    np_test_ds = NPWrapperDataset(torch_dataset=test_ds, add_X_dimension=add_X_dimension)

    return np_train_ds, np_test_ds


def get_synthetic_presplit_dataset(dataset_path: str, first_observation_idx: int, amount_of_observations: int,
                                   gt_key: str, add_X_dimension: bool):
    """
    Returns a numpy version of an already split and saved synthetic dataset

    Parameters
    ----------
    dataset_path : str
                Path to a synthetic dataset
    first_observation_idx : int
                Index of a first observation (time-series) to consider. For example, if first_observation_idx == 5
                it will ignore time-series with indices 0, 1, 2, 3, 4
    amount_of_observations : int
                Amount of observations (time-series) to take from the pre-split dataset. For example, with
                first_observation_idx == 5 and amount_of_observations == 10 it will take observations 5 to 5+10 from
                the pre-split dataset. Important: first_observation_idx + amount_of_observations cannot exceed the
                amount of observations in the pre-split dataset
    gt_key : str
                Key to the ground truth for the synthetic data. Note, in pre-split synthetic datasets we have keys
                True_GT and HFIR_GT. True_GT will give an actual noiseless analytic signal from which noisy
                samples were generated. HFIR_GT gives an approximation of True_GT by filtering noisy samples with
                HFIR filter
    add_X_dimension : bool
                Whether to add additional dimension to the input time-series. If False input time-series shape
                is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications

    Returns
    -------
    np_ds : NPWrapperDataset
                Numpy version of the required synthetic dataset
    """
    torch_ds = PreSplitDataset(
        dataset_path=dataset_path,
        first_observation_idx=first_observation_idx,
        amount_of_observations=amount_of_observations,
        input_size=5024,
        output_size=4000,
        gt_key=gt_key,
        split_ds_into_chunks=False,
        input_chunk_length=None,
        output_chunk_length=None
    )

    np_ds = NPWrapperDataset(torch_dataset=torch_ds, add_X_dimension=add_X_dimension)

    return np_ds


def get_synthetic_generated_dataset(dataset_name: str, frequency: float, duration: float, sampling_rate: float,
                                    filter_band: tp.Optional[tuple[float, float]], noise_coeff: float,
                                    amount_of_observations: int, add_X_dimension: bool):
    """
    Returns a numpy version of a generated synthetic dataset

    Parameters
    ----------
    dataset_name : str
                Synthetic dataset name to generate for test. Should be in ['sines_white', 'sines_pink',
                'filtered_pink', 'state_space_white', 'state_space_pink']
    frequency : float
                Central frequency for test dataset generation. Use None for filtered_pink as it relies on filter_band
    duration : float
                Amount of seconds per single time-series in a test dataset
    sampling_rate : float
                Sampling rate for the test data
    filter_band : tuple (band_low, band_high)
                Band to filter ground truth in for the filtered_pink test dataset. Use None for others
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified for testing dataset. Used to calculate Noise
                level. For white noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    amount_of_observations : int
                Amount of time-series (observations) in a resultant test dataset. Total dataset size is
                duration * fs * amount_of_observations samples. Each time-series will be generated independently
    add_X_dimension : bool
                Whether to add additional dimension to the input time-series. If False input time-series shape
                is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications

    Returns
    -------
    np_ds : NPWrapperDataset
                Numpy version of the required synthetic dataset
    """
    input_chunk_length = 1000  # input_chunk_length from the neural net experiments for consistency during generation
    torch_ds = GeneratedSyntheticDataset(
        duration=duration + 1 + math.ceil(input_chunk_length / sampling_rate),
        fs=sampling_rate,
        dataset_name=dataset_name,
        frequency=frequency,
        filter_band=filter_band,
        input_size=int(duration * sampling_rate + input_chunk_length),
        output_size=int(duration * sampling_rate),
        amount_of_observations=amount_of_observations,
        noise_coeff=noise_coeff,
        rng_seed=0
    )

    np_ds = NPWrapperDataset(torch_dataset=torch_ds, add_X_dimension=add_X_dimension)

    return np_ds


def get_synthetic_datasets(train_dataset_path: str, train_first_observation_idx: int, train_amount_of_observations: int,
                           train_gt_key: str,
                           test_dataset_name: str, test_frequency: float,
                           test_duration: float, test_sampling_rate: float,
                           test_data_filter_band: tp.Optional[tuple[float, float]], test_amount_of_observations: int,
                           test_noise_coeff: float, add_X_dimension: bool):
    """
    Get a pre-split and generated synthetic datasets for training and testing respectively

    Parameters
    ----------
    train_dataset_path : str
                Path to a synthetic training dataset
    train_first_observation_idx : int
                Index of a first observation (time-series) in training dataset to consider.
                For example, if first_observation_idx == 5 it will ignore time-series with indices 0, 1, 2, 3, 4
    train_amount_of_observations : int
                Amount of observations (time-series) to take from the pre-split training dataset. For example, with
                first_observation_idx == 5 and amount_of_observations == 10 it will take observations 5 to 5+10 from
                the pre-split dataset. Important: first_observation_idx + amount_of_observations cannot exceed the
                amount of observations in the pre-split dataset
    train_gt_key : str
                Key to the ground truth for the synthetic training data. Note, in pre-split synthetic datasets we have
                keys True_GT and HFIR_GT. True_GT will give an actual noiseless analytic signal from which noisy
                samples were generated. HFIR_GT gives an approximation of True_GT by filtering noisy samples with
                HFIR filter
    test_dataset_name : str
                Synthetic dataset name to generate for test. Should be in ['sines_white', 'sines_pink',
                'filtered_pink', 'state_space_white', 'state_space_pink']
    test_frequency : float
                Central frequency for test dataset generation. Use None for filtered_pink as it relies on filter_band
    test_duration : float
                Amount of seconds per single time-series in a test dataset
    test_sampling_rate : float
                Sampling rate for the test data
    test_data_filter_band : tuple (band_low, band_high)
                Band to filter ground truth in for the filtered_pink test dataset. Use None for others
    test_amount_of_observations : int
                Amount of time-series (observations) in a resultant test dataset. Total dataset size is
                duration * fs * amount_of_observations samples. Each time-series will be generated independently
    test_noise_coeff : float
                Noise coefficient by which a sample of noise is amplified for testing dataset. Used to calculate Noise
                level. For white noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    add_X_dimension : bool
                Whether to add additional dimension to the input time-series. If False input time-series shape
                is (duration * fs, ). If True (duration * fs, 1). Can be useful for matrix multiplications

    Returns
    -------
    np_train_ds : NPWrapperDataset
                Numpy version of PreSplitDataset for training
    np_test_ds : NPGeneratedDataset
                Numpy generated dataset for testing
    """
    train_ds = get_synthetic_presplit_dataset(dataset_path=train_dataset_path,
                                              first_observation_idx=train_first_observation_idx,
                                              amount_of_observations=train_amount_of_observations,
                                              gt_key=train_gt_key,
                                              add_X_dimension=add_X_dimension)
    test_ds = get_synthetic_generated_dataset(dataset_name=test_dataset_name, frequency=test_frequency,
                                              duration=test_duration, sampling_rate=test_sampling_rate,
                                              filter_band=test_data_filter_band, noise_coeff=test_noise_coeff,
                                              amount_of_observations=test_amount_of_observations,
                                              add_X_dimension=add_X_dimension)

    return train_ds, test_ds

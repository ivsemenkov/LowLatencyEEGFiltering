import warnings
import numpy as np
import torch
from pytorch_lightning import LightningModule
from general_utilities.metrics import calculate_batched_circstd, calculate_batched_correlation, calculate_batched_plv
from general_utilities.utils import calculate_statistics


class EEGFilteringSystem(LightningModule):
    def __init__(self,
                 eeg_filtering_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion: torch.nn.Module,
                 save_predictions: bool = False
                 ):
        """
        Pytorch Lightning model

        Parameters
        ----------
        eeg_filtering_model : torch.nn.Module
                    Original PyTorch filtering network
        optimizer : torch.optim.Optimizer
                    Initialized optimizer
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
                    Initialized instance of ReduceLROnPlateau learning rate scheduler
        criterion : torch.nn.Module
                    Initialized loss function
        save_predictions : bool
                    If True, keeps track of predictions and ground truth for different batches. Later can be saved
        """
        super().__init__()

        self.criterion = criterion
        self.eeg_filtering_model = eeg_filtering_model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.metric_names = ['correlation', 'plv', 'circstd', 'circstd_degrees']

        self.reset_test_metrics()
        self.reset_predictions()
        self.save_predictions = save_predictions
        self.warned = False

    def forward(self, batch_data: tuple[torch.Tensor, torch.Tensor]):
        """
        Processed a single chunk of observations

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch

        Returns
        -------
        outputs : torch.Tensor
                    Tensor with models output
        """
        inputs, _ = batch_data
        inputs_size = inputs.size(1)
        input_chunk_length = self.eeg_filtering_model.input_chunk_length
        assert inputs_size == input_chunk_length, (inputs_size, input_chunk_length)

        model_out = self.eeg_filtering_model(inputs)
        if model_out.dim() == 4:
            if model_out.size(-1) > 1 and not self.warned:
                warnings.warn(f'Likelihood parameter is {model_out.size(-1)}. '
                              f'Careful as only the first dimension is used')
                self.warned = True
            outputs = model_out[..., 0]
        else:
            assert model_out.dim() == 3
            outputs = model_out
        return outputs

    def filter_full_timeseries(self, batch_data: tuple[torch.Tensor, torch.Tensor]):
        """
        Processed a whole time-series of observations one sample at a time imitating real-time framework

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth time-series for the same batch

        Returns
        -------
        outputs : torch.Tensor
                    Tensor with models output time-series
        """
        inputs, gt = batch_data
        output_size = gt.size(1)
        inputs_size = inputs.size(1)
        input_chunk_length = self.eeg_filtering_model.input_chunk_length
        start_idx = inputs_size - output_size - input_chunk_length + 1
        assert start_idx >= 0

        batch_size = inputs.size(0)
        outputs = torch.empty((batch_size, output_size, 2), dtype=torch.float32, device=self.device)

        for step_idx in range(output_size):

            current_idx = start_idx + step_idx
            new_points = inputs[:, current_idx:current_idx+input_chunk_length, :]
            model_out = self.eeg_filtering_model(new_points)
            if model_out.dim() == 4:
                if model_out.size(-1) > 1 and not self.warned:
                    warnings.warn(f'Likelihood parameter is {model_out.size(-1)}. '
                                  f'Careful as only the first dimension is used')
                    self.warned = True
                outputs[:, step_idx, :] = model_out[:, -1, :, 0]
            else:
                assert model_out.dim() == 3
                outputs[:, step_idx, :] = model_out[:, -1, :]
        return outputs

    def log_errors(self, filtering_loss: torch.Tensor, errors_dict: dict, tag: str):
        """
        Log errors while training, validating or testing

        Parameters
        ----------
        filtering_loss : torch.Tensor
                    Loss value for the batch
        errors_dict : dict
                    Dictionary which has aggregated error values for appropriate metrics
        tag : str
                    Tag of a current errors (train, val  or test)
        """
        self.log(f'{tag}_loss', filtering_loss)
        for metric_name in self.metric_names:
            self.log(f'{tag}_mean_{metric_name}', errors_dict[f'mean_{metric_name}'])

    def common_step(self, batch_data: tuple[torch.Tensor, torch.Tensor], mode: str):
        """
        Processing step which is common for training, validation and test

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch
        mode : str
                    Current mode: train, val  or test

        Returns
        -------
        outputs : torch.Tensor
                    Tensor with models output
        filtering_loss : torch.Tensor
                    Loss value for the batch
        filtering_errors_dict : dict
                    Dictionary which has aggregated error values for appropriate metrics
        detailed_errors_dict : dict
                    Dictionary which has lists of error for appropriate metrics
                    ['correlation', 'plv', 'circstd', 'circstd_degrees']
        """
        if mode == 'test':
            outputs = self.filter_full_timeseries(batch_data)
        else:
            outputs = self.forward(batch_data)

        outputs, filtering_loss = self.compute_losses(batch_data, outputs)
        filtering_errors_dict, detailed_errors_dict = self.compute_errors(batch_data, outputs)
        return outputs, filtering_loss, filtering_errors_dict, detailed_errors_dict

    def training_step(self, batch_data: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a single training step

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch
        batch_idx : int
                    Index of a batch

        Returns
        -------
        filtering_loss : torch.Tensor
                    Loss value for the batch
        """
        outputs, filtering_loss, filtering_errors_dict, _ = self.common_step(batch_data, 'train')
        self.log_errors(filtering_loss=filtering_loss, errors_dict=filtering_errors_dict, tag='train')

        return filtering_loss

    def validation_step(self, batch_data: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a single validation step

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch
        batch_idx : int
                    Index of a batch

        Returns
        -------
        filtering_loss : torch.Tensor
                    Loss value for the batch
        """
        outputs, filtering_loss, filtering_errors_dict, _ = self.common_step(batch_data, 'val')
        self.log_errors(filtering_loss=filtering_loss, errors_dict=filtering_errors_dict, tag='val')

        return filtering_loss

    def test_step(self, batch_data: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a single testting step

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch
        batch_idx : int
                    Index of a batch

        Returns
        -------
        filtering_errors_dict : dict
                    Dictionary which has aggregated error values for appropriate metrics
        """
        outputs, filtering_loss, filtering_errors_dict, detailed_errors_dict = self.common_step(batch_data, 'test')
        self.log_errors(filtering_loss=filtering_loss, errors_dict=filtering_errors_dict, tag='test')

        filtering_errors_dict['total_loss_value'] = filtering_loss
        self.add_test_metrics(filtering_loss.item(), filtering_errors_dict, detailed_errors_dict)

        return filtering_errors_dict

    def reset_test_metrics(self):
        """
        Resets all saved during testing metrics
        """
        self.test_metrics = {'loss_values': [],
                             'errors_per_batch_reduced_dict': {f'{reduce_function}_{metric_name}': [] for
                                                               reduce_function in ['min', 'max', 'mean', 'median']
                                                               for metric_name in self.metric_names},
                             'errors_detailed_dict': {metric_name: [] for metric_name in self.metric_names}}

    def set_save_predictions(self, value: bool):
        """
        Set value for self.save_predictions

        Parameters
        ----------
        value : bool
                    If True, keeps track of predictions and ground truth for different batches. Later can be saved
        """
        self.save_predictions = value

    def add_test_metrics(self, filtering_loss: float, filtering_errors_dict: dict, detailed_errors_dict: dict):
        """
        Add new metrics to all test metrics

        Parameters
        ----------
        filtering_loss : float
                    Loss value for the batch cast to float
        filtering_errors_dict : dict
                    Dictionary which has aggregated error values for appropriate metrics
        detailed_errors_dict : dict
                    Dictionary which has lists of error for appropriate metrics
                    ['correlation', 'plv', 'circstd', 'circstd_degrees']
        """
        self.test_metrics['loss_values'].append(filtering_loss)
        for metric_name in self.metric_names:
            metric_values = detailed_errors_dict[metric_name].detach().cpu().numpy().tolist()
            self.test_metrics['errors_detailed_dict'][metric_name].extend(metric_values)
            for reduce_function in ['min', 'max', 'mean', 'median']:
                dict_key = f'{reduce_function}_{metric_name}'
                metric_value = filtering_errors_dict[dict_key]
                self.test_metrics['errors_per_batch_reduced_dict'][dict_key].append(metric_value)

    def configure_optimizers(self):
        """
        Configures optimizer and scheduler

        Returns
        -------
        optimizers : list
                    List of optimizers
        schedulers : list
                    List of schedulers
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {'scheduler': self.scheduler, 'monitor': 'val_loss'}
        else:
            scheduler = self.scheduler
        return [self.optimizer], [scheduler]

    def compute_losses(self, batch_data: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor):
        """
        Compute loss values

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch
        outputs : torch.Tensor
                    Tensor with models output

        Returns
        -------
        outputs : torch.Tensor
                    Tensor with models output
        filtering_loss : torch.Tensor
                    Loss value for the batch
        """
        _, gt = batch_data

        return outputs, self.criterion(outputs, gt)

    def add_predictions(self, complex_pred: torch.Tensor, complex_gt: torch.Tensor, envelope_pred: torch.Tensor,
                        envelope_gt: torch.Tensor):
        """
        Add new predictions and ground truth to the all saved predictions and ground truth

        Parameters
        ----------
        complex_pred : torch.Tensor
                    Complex valued predicted filtered analytic signal
        complex_gt : torch.Tensor
                    Complex valued ground truth analytic signal
        envelope_pred : torch.Tensor
                    Envelope of a predicted filtered analytic signal
        envelope_gt : torch.Tensor
                    Envelope of a ground truth analytic signal
        """
        self.predictions['complex_pred'].extend(complex_pred.detach().cpu().numpy())
        self.predictions['complex_gt'].extend(complex_gt.detach().cpu().numpy())
        self.predictions['envelope_pred'].extend(envelope_pred.detach().cpu().numpy())
        self.predictions['envelope_gt'].extend(envelope_gt.detach().cpu().numpy())

    def save_predictions_npz(self, filename: str):
        """
        Save all predictions and ground truth

        Parameters
        ----------
        filename : str
                    Path to the npz file for writing
        """
        np.savez(filename, **self.predictions)

    def reset_predictions(self):
        """
        Resets all currently saved predictions, ground truth
        """
        self.predictions = {'complex_pred': [],
                            'complex_gt': [],
                            'envelope_pred': [],
                            'envelope_gt': []}

    def compute_errors(self, batch_data: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor):
        """
        Compute all metrics for batch

        Parameters
        ----------
        batch_data : tuple
                    Tuple consisting of inputs and ground truth for the same batch
        outputs : torch.Tensor
                    Tensor with models output

        Returns
        -------
        errors_dict : dict
                    Dictionary which has aggregated error values for appropriate metrics
        detailed_errors_dict : dict
                    Dictionary which has lists of error for appropriate metrics
                    ['correlation', 'plv', 'circstd', 'circstd_degrees']
        """
        _, gt = batch_data

        complex_pred = outputs[:, :, 0] + 1j * outputs[:, :, 1]
        complex_gt = gt[:, :, 0] + 1j * gt[:, :, 1]

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
            'correlation': correlations_per_batch,
            'plv': plvs_per_batch,
            'circstd': circstd_per_batch,
            'circstd_degrees': circstd_per_batch_degrees
        }

        if self.save_predictions:
            self.add_predictions(complex_pred, complex_gt, envelope_pred, envelope_gt)

        return errors_dict, detailed_errors_dict

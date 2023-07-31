"""
Temporal Convolutional Network
------------------------------
Modified implementation from DARTS python library:
https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tcn_model.py

"""
import math
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from darts.utils.torch import MonteCarloDropout


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_channel_size: int,
        target_size: int,
    ):
        """
        PyTorch module implementing a residual block module

        Parameters
        ----------
        num_filters : int
                    The number of filters in a convolutional layer of the TCN
        kernel_size : int
                    The size of every kernel in a convolutional layer
        dilation_base : int
                    The base of the exponent that will determine the dilation on every level
        dropout_fn :
                    The dropout function to be applied to every convolutional layer
        weight_norm : bool
                    Boolean value indicating whether to use weight normalization
        nr_blocks_below : int
                    The number of residual blocks before the current one
        num_layers : int
                    The number of convolutional layers
        input_channel_size : int
                    The dimensionality of the input time series of the whole network
        target_size : int
                    The dimensionality of the output time series of the whole network
        """
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_channel_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        """
        Process batch

        Parameters
        ----------
        x : torch.Tensor
                    Input tensor of size (batch_size, in_dimension, input_chunk_length)

        Returns
        -------
        y : torch.Tensor
                    Output tensor of size (batch_size, out_dimension, input_chunk_length)
        """
        residual = x

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class TCNFilteringModel(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        input_channel_size: int,
        output_chunk_length: int,
        kernel_size: int,
        num_filters: int,
        num_layers: Optional[int],
        dilation_base: int,
        weight_norm: bool,
        target_size: int,
        nr_params: int,
        dropout: float
    ):
        """
        PyTorch module implementing a dilated TCN module

        Parameters
        ----------
        input_chunk_length : int
                    Amount of lags the model should take as a single input
        input_channel_size : int
                    The dimensionality of the input time series.
        output_chunk_length : int
                    Amount of final observations for which model should output filtered analytic signal
        kernel_size : int
                    The size of every kernel in a convolutional layer
        num_filters : int
                    The number of filters in a convolutional layer of the TCN
        num_layers : int
                    The number of convolutional layers
        dilation_base : int
                    The base of the exponent that will determine the dilation on every level
        weight_norm : bool
                    Boolean value indicating whether to use weight normalization
        target_size : int
                    The dimensionality of the output time series
        nr_params : int
                    The number of parameters of the likelihood (or 1 if no likelihood is used)
        dropout : float
                    The dropout rate for every convolutional layer
        """
        super().__init__()

        # Additional parameters
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # Defining parameters
        self.input_channel_size = input_channel_size
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        # self.target_length = target_length
        self.target_size = target_size
        self.nr_params = nr_params
        self.dilation_base = dilation_base
        self.dropout = MonteCarloDropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(
                math.log(
                    (self.input_chunk_length - 1)
                    * (dilation_base - 1)
                    / (kernel_size - 1)
                    / 2
                    + 1,
                    dilation_base,
                )
            )
            # logger.info("Number of layers chosen: " + str(num_layers))
        elif num_layers is None:
            num_layers = math.ceil(
                (self.input_chunk_length - 1) / (kernel_size - 1) / 2
            )
            # logger.info("Number of layers chosen: " + str(num_layers))
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_channel_size,
                target_size * nr_params,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        """
        Process batch

        Parameters
        ----------
        x : torch.Tensor
                    Input tensor of size (batch_size, input_chunk_length, input_channel_size)

        Returns
        -------
        y : torch.Tensor
                    Output tensor of size (batch_size, output_chunk_length, target_size, nr_params)
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.transpose(1, 2)
        x = x.view(
            batch_size, self.input_chunk_length, self.target_size, self.nr_params
        )

        return x[:, -self.output_chunk_length:, :, :]

import torch.nn as nn
import torchaudio


class ConvTasNetFilteringModel(nn.Module):
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 num_sources: int = 2):
        """
        Initialize Conv-TasNet model

        Parameters
        ----------
        input_chunk_length : int
                    Amount of lags the model should take as a single input
        output_chunk_length : int
                    Amount of final observations for which model should output filtered analytic signal
        num_sources : int
                    Amount of output dimensions
        """
        super().__init__()
        self.torchaudio_module = torchaudio.models.ConvTasNet(num_sources=num_sources,
                                                              enc_kernel_size=16,
                                                              enc_num_feats=512,
                                                              msk_kernel_size=3,
                                                              msk_num_feats=128,
                                                              msk_num_hidden_feats=512,
                                                              msk_num_layers=8,
                                                              msk_num_stacks=3,
                                                              msk_activate="relu")
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    def forward(self, x):
        """
        Filter a series

        Parameters
        ----------
        x : torch.Tensor
                    Noisy observations. Size [batch_size, input_chunk_length, 1]

        Returns
        -------
        prediction : torch.Tensor
                    Filtered final samples. Size [batch_size, output_chunk_length, num_sources]
        """

        x = x.transpose(1, 2)
        x = self.torchaudio_module(x).transpose(1, 2)

        return x[:, -self.output_chunk_length:, :]

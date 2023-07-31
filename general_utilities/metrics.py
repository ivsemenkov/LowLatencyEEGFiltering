import torch


def calculate_batched_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates correlation per sample for a batch

    Parameters
    ----------
    x : torch.Tensor
                Real-valued predictions of size (batch_size, time_steps)
    y : torch.Tensor
                Real-valued ground truth of size (batch_size, time_steps)

    Returns
    -------
    correlations : torch.Tensor
                Correlation values per batch element of size (batch_size, )
    """
    means_x = torch.mean(x, dim=1, keepdim=True)
    means_y = torch.mean(y, dim=1, keepdim=True)
    diff_x = x - means_x
    diff_y = y - means_y
    x_std = torch.sqrt(torch.sum(diff_x ** 2, dim=1, keepdim=True))
    y_std = torch.sqrt(torch.sum(diff_y ** 2, dim=1, keepdim=True))
    cov = torch.sum(diff_x * diff_y, dim=1, keepdim=True)
    correlations = (cov / (x_std * y_std))
    assert correlations.size(1) == 1
    correlations = correlations[:, 0]  # always shape (BATCH_SIZE, 1)
    return correlations


def calculate_batched_plv(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates Phase Locking Value (PLV) per sample for a batch

    Parameters
    ----------
    x : torch.Tensor
                Complex-valued predictions of size (batch_size, time_steps)
    y : torch.Tensor
                Complex-valued ground truth of size (batch_size, time_steps)

    Returns
    -------
    plv_values : torch.Tensor
                PLV values per batch element of size (batch_size, )
    """
    assert x.size() == y.size()
    normed_x = x / torch.abs(x)
    normed_y = y / torch.abs(y)
    plv_values = torch.abs(torch.mean(torch.conj(normed_y) * normed_x, dim=1))
    return plv_values


def calculate_batched_circstd(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates Circular standard deviation per sample for a batch

    Parameters
    ----------
    x : torch.Tensor
                Complex-valued predictions of size (batch_size, time_steps)
    y : torch.Tensor
                Complex-valued ground truth of size (batch_size, time_steps)

    Returns
    -------
    circstd_values : torch.Tensor
                CircSTD values per batch element of size (batch_size, )
    """
    assert x.size() == y.size()
    x_angle = x.angle()
    y_angle = y.angle()
    means = torch.abs(torch.mean(torch.exp(1.j * (y_angle - x_angle)), dim=1))
    result = torch.sqrt(-2. * torch.log(means))
    return result

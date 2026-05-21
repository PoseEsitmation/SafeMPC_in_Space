import torch

def to_device(x, device):
    """
    Move a tensor or tensor-like object to the specified device.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be moved.
    device : str or torch.device
        Target device, e.g. "cpu", "cuda:0", "mps".

    Returns
    -------
    torch.Tensor
        Tensor located on the target device.

    Example
    -------
    x = torch.randn(3)
    x = to_device(x, "cuda:0")
    """
    return x.to(device)


def tensor(data, device, **kwargs):
    """
    Create a torch tensor directly on a specified device.

    This avoids the common anti-pattern:
        torch.tensor(...).to(device)

    and ensures device placement happens at creation time.

    Parameters
    ----------
    data : array-like
        Input data (list, numpy array, scalar, etc.).
    device : str or torch.device
        Target device.
    **kwargs :
        Additional arguments passed to torch.tensor.

    Returns
    -------
    torch.Tensor
        Tensor placed directly on the specified device.

    Example
    -------
    x = tensor([1, 2, 3], device="cuda:0", dtype=torch.float32)
    """
    return torch.tensor(data, device=device, **kwargs)


def zeros(*shape, device, **kwargs):
    """
    Create a zero-initialized tensor on a specific device.

    Replaces:
        torch.zeros(...).to(device)

    Ensures allocation happens directly on the correct device.

    Parameters
    ----------
    *shape :
        Shape of the tensor (e.g. 3, 4 -> (3, 4)).
    device : str or torch.device
        Target device.
    **kwargs :
        Additional arguments passed to torch.zeros.

    Returns
    -------
    torch.Tensor
        Zero tensor on the specified device.

    Example
    -------
    z = zeros(2, 5, device="cpu")
    """
    return torch.zeros(*shape, device=device, **kwargs)


def randn(*shape, device, **kwargs):
    """
    Create a tensor with values sampled from a normal distribution on a device.

    Replaces:
        torch.randn(...).to(device)

    Useful for noise generation in:
    - MPPI
    - exploration policies
    - stochastic dynamics

    Parameters
    ----------
    *shape :
        Shape of the tensor.
    device : str or torch.device
        Target device.
    **kwargs :
        Additional arguments passed to torch.randn.

    Returns
    -------
    torch.Tensor
        Random normal tensor on the specified device.

    Example
    -------
    eps = randn(32, 10, device="cuda:0")
    """
    return torch.randn(*shape, device=device, **kwargs)
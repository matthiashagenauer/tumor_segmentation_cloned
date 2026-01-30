from typing import Any, List, Mapping, Optional

import torch


class AdditativeLR(torch.optim.lr_scheduler._LRScheduler):
    """Change the current step length by adding 'addend' every 'update_frequency'
    steps. 'addend' is positive for increasing step length and negative for decreasing.

    Notice that such decay can happen simultaneously with other changes to the learning
    rate from outside this scheduler. When last_epoch=-1, sets initial step length as
    step length.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        update_frequency (int): Number of steps before changing the step length.
        addend (float): Number that is added to the current step length.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer use initial step length = 0.1 for all groups
        >>> # lr = 0.1    if epoch < 30
        >>> # lr = 0.2    if 30 <= epoch < 60
        >>> # lr = 0.3    if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = AdditativeLR(optimizer, update_frequency=30, addend=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        update_frequency: int,
        addend: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.update_frequency = update_frequency
        self.addend = addend
        super(AdditativeLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if (self.last_epoch == 0) or (self.last_epoch % self.update_frequency != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] + self.addend for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> List[float]:
        return [
            base_lr + self.addend * (self.last_epoch // self.update_frequency)
            for base_lr in self.base_lrs
        ]


def get_schedule(
    optimiser: torch.optim.Optimizer,
    schedule: Optional[str],
    config: Mapping[str, Any],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if schedule is None:
        return None
    if schedule == "constant":
        return None
    elif schedule == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimiser, **config)
    elif schedule == "additative":
        return AdditativeLR(optimiser, **config)
    elif schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimiser, **config)
    elif schedule == "cosine":
        assert "t_max" in config.keys()
        t_max = config["t_max"]
        # Remove t_max from config
        config = {k: v for k, v in config.items() if k != "t_max"}
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=t_max, **config
        )
    else:
        raise ValueError(f"Schedule `{schedule}` not implemented yet")

import torch


class AdaptiveTrainingScheduler:
    """Adaptive Training Scheduler that supports multiple scheduling strategies.

    Supported scheduler types:
        - 'plateau': Uses torch.optim.lr_scheduler.ReduceLROnPlateau. Metric value is required for step().
        - 'cosine': Uses torch.optim.lr_scheduler.CosineAnnealingWarmRestarts with epoch-based step.

    Example usage:

        scheduler = AdaptiveTrainingScheduler(optimizer, scheduler_type='plateau', mode='min',
                                                factor=0.5, patience=2, verbose=True)
        for epoch in range(num_epochs):
            train_loss = ...
            scheduler.step(epoch, metric=train_loss)

        # For cosine:
        scheduler = AdaptiveTrainingScheduler(optimizer, scheduler_type='cosine', T_0=10, T_mult=1, eta_min=1e-7)
        for epoch in range(num_epochs):
            scheduler.step(epoch)
    """
    def __init__(self, optimizer, scheduler_type='plateau', **kwargs):
        self.type = scheduler_type
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
        else:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")

    def step(self, epoch, metric=None):
        if self.type == 'plateau':
            if metric is None:
                raise ValueError("For 'plateau' scheduler, metric must be provided in step().")
            self.scheduler.step(metric)
        elif self.type == 'cosine':
            self.scheduler.step(epoch)
        else:
            raise ValueError(f"Scheduler type '{self.type}' is not supported.")

    def get_last_lr(self):
        """Return last learning rate from the underlying scheduler."""
        return self.scheduler.get_last_lr() 
import torch
from torch.optim.optimizer import Optimizer, required


class SGLD(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0.9,
                 dampening=0,
                 weight_decay=0.0001):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=False)
        super(SGLD, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.params = params
    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # for p in self.world_model.parameters():
        #     buf_new = (1 - momentum) * p.buf - lr * d_p
        #     # Noise
        #     eps = torch.randn(p.size())
        #     buf_new += (2.0 * lr * 0.9 * temperature / datasize) ** .5 * eps
        #     p.data.add_(buf_new)
        #     p.buf = buf_new

        for p in self.params.parameters():
            if not hasattr(p, 'buf'):
                p.buf = torch.zeros(p.size())
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1 - self.momentum) * p.buf - self.lr * d_p
            eps = torch.randn(p.size())
            buf_new += (2.0 * self.lr * self.momentum * (1e-7)) ** .5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

        # for p in net.parameters():
        #     p.data.add_(buf_new)
        #     p.buf = buf_new

        return 1.0
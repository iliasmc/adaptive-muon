"""
The code for muon with probabilistic scheduler.
"""

import numpy as np
import torch


def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)

        self.current_step = 0
        self.current_epoch = 0
        self.cycle_size = 3
        self.prob_orth = 0.9
        self.decay_rate = 0.069
        self.step_size = 2
        self.scheduler = "uniform"
        self.pr_min = 0
        self.pr_max = 0.8
        self.orthogonalization_count = 0
        self.skipped_orthogonalization_count = 0
        np.random.seed(1)

        super().__init__(params, defaults)
    
    def update_epoch(self):
        self.current_epoch = self.current_epoch + 1
    
    def step(self):
        self.current_step += 1
        
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            if self.scheduler == "uniform":
                do_orth = bool(np.random.binomial(1, self.prob_orth))
            elif self.scheduler == "exponential":
                # best combination for now decay_rate = 0.0038 and prob_orth = 1.0
                prob_orth = self.prob_orth* np.exp(-self.decay_rate * self.current_epoch)
                do_orth = bool(np.random.binomial(1, prob_orth))
            elif self.scheduler == "step_decay":
                prob_orth = self.prob_orth * (self.decay_rate ** np.floor(self.current_epoch/self.step_size))
                do_orth = bool(np.random.binomial(1, prob_orth))
            elif self.scheduler == "cosine_annealing":
                prob_orth = self.pr_min + 0.5 * (self.pr_max - self.pr_min) * (1 + np.cos(self.current_epoch/ self.cycle_size * np.pi))
                do_orth = bool(np.random.binomial(1, prob_orth))
            elif self.scheduler == "cyclic":
                cycle = np.floor( 1 + self.current_step/(2*self.stepsize))
                x = abs(self.current_step/self.stepsize - 2*cycle + 1)
                new_pr = self.pr_min + (self.pr_max - self.pr_min) * max(0, 1-x)
                do_orth = bool(np.random.binomial(1, new_pr))
            else:
                print(f"The picked scheduler: {self.scheduler} has not been implemented!")
                return 
            # print("THIS IS A STEP!") -> once 
            for p in group["params"]:
                # print("THIS IS A GROUP!") -> six times
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())  # normalize the weight

                # Periodic orthogonalization
                if do_orth or "buffer" not in state:
                    if "buffer" not in state:
                        state["buffer"] = torch.zeros_like(g)

                    # compute new orthogonalized update from momentum m
                    update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(
                        g.shape
                    )  # whiten the update
                    state["buffer"].copy_(update)
                    self.orthogonalization_count += 1
                    # p.data.mul_(len(p.data) ** 0.5 / p.data.norm())  # normalize the weight
                    p.data.add_(update, alpha=-lr)  # take a step
                else:
                    # update = g
                    self.skipped_orthogonalization_count += 1
                    p.data.add_(g, alpha=-lr)
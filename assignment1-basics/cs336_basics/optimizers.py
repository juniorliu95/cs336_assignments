"""
optimizers for cs336 assignment1 test and for further reuse
junyu, 2026.10.26
"""

import torch



class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        """
        Implements the AdamW optimizer. 
        Params directly copied from torch.optim.AdamW.

        Args:
            params (list[torch.nn.Parameter]): List of parameters to optimize.
            lr (float): Learning rate.
            betas (tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            weight_decay (float): Weight decay (L2 penalty).
            amsgrad (bool): Whether to use the AMSGrad variant of this algorithm.
            maximize (bool): Whether to maximize the objective function.
        """

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        logic w.r.t. https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html,
        order different from the handout
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if state.get("m") is None:
                    state["m"] = torch.zeros_like(param.data)
                if state.get("v") is None:
                    state["v"] = torch.zeros_like(param.data)
                if state.get("count") is None:
                    state["count"] = 1

                gt = None
                if maximize:
                    gt = -param.grad.data
                else:
                    gt = param.grad.data

                param.data = param.data - lr * weight_decay * param.data

                state["m"] = betas[0] * state["m"] + (1 - betas[0]) * gt
                state["v"] = betas[1] * state["v"] + (1 - betas[1]) * gt * gt
                m_hat = state["m"] / (1 - betas[0] ** state["count"])

                v_hat = None
                if amsgrad:
                    if state.get("v_max") is None:
                        state["v_max"] = state["v"]
                    else: 
                        state["v_max"] = torch.max(state["v"], state["v_max"])
                    v_hat = state["v_max"] / (1 - betas[1] ** state["count"])
                else:
                    v_hat = state["v"] / (1 - betas[1] ** state["count"])

                param.data = param.data - lr * m_hat / (torch.sqrt(v_hat) + eps)
            
                state["count"] += 1
        return loss


            

    

   
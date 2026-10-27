"""The reflection model: one signed RGB image R shared by every training frame.

For training frame t, with T_t the 3DGS render:

    during the warm-up (iteration <= t_0):   I_t ~ T_t
    afterwards:                              I_t ~ (1 - A) * a_t(T_t) + W * R

W is the fixed self-localization weight (pass 2 only; all ones in pass 1). Two terms are optional:
A, a per-pixel attenuation in [0, atten_max), and a_t, a per-frame RGB gain and bias that absorbs
exposure and white-balance drift between frames. The configuration reported in the paper uses both in
pass 2 (see README). Held-out views render T only: R, A and a_t never enter evaluation.
"""
import os

import torch


class ReflectionModel:
    def __init__(self, height, width, frame_names, device="cuda", lr=0.01, l1=0.01, weight=None,
                 atten=False, atten_max=0.9, atten_l1=0.01,
                 appearance="none", appearance_lr=1e-3, appearance_reg=10.0):
        if appearance not in ("none", "affine"):
            raise ValueError(f"appearance must be 'none' or 'affine', got {appearance!r}")
        self.l1 = l1
        self.atten_max = atten_max
        self.atten_l1 = atten_l1
        self.appearance = appearance
        self.appearance_reg = appearance_reg
        self.weight = None if weight is None else weight.to(device)        # [1,H,W] in [0,1]

        self.R = torch.nn.Parameter(torch.zeros(3, height, width, device=device))
        groups = [{"params": [self.R], "lr": lr}]
        self.atten_logit = None
        if atten:
            # A = atten_max * sigmoid(-3) ~ 0.04 at the start: nearly no attenuation.
            self.atten_logit = torch.nn.Parameter(torch.full((1, height, width), -3.0, device=device))
            groups.append({"params": [self.atten_logit], "lr": lr})
        self.gain, self.bias = {}, {}
        if appearance == "affine":
            for name in frame_names:
                self.gain[name] = torch.nn.Parameter(torch.ones(3, 1, 1, device=device))
                self.bias[name] = torch.nn.Parameter(torch.zeros(3, 1, 1, device=device))
                groups.append({"params": [self.gain[name], self.bias[name]], "lr": appearance_lr})
        self.optimizer = torch.optim.Adam(groups)

    def attenuation(self):
        if self.atten_logit is None:
            return None
        return self.atten_max * torch.sigmoid(self.atten_logit)

    def predict(self, T, name):
        """Predicted observation of training frame `name` from its render T [3,H,W].

        Returns (prediction, appearance regularizer for this frame).
        """
        R = self.R if self.weight is None else self.R * self.weight
        reg = T.new_zeros(())
        if self.appearance == "affine":
            gain, bias = self.gain[name], self.bias[name]
            T = gain * T + bias
            reg = ((gain - 1.0) ** 2).mean() + (bias ** 2).mean()
        A = self.attenuation()
        if A is not None:
            T = (1.0 - A) * T
        return T + R, reg

    def penalty(self, appearance_reg):
        """Regularizers added to the photometric loss once R is active."""
        loss = self.l1 * self.R.abs().mean()
        if self.appearance != "none":
            loss = loss + self.appearance_reg * appearance_reg
        A = self.attenuation()
        if A is not None:
            loss = loss + self.atten_l1 * A.mean()
        return loss

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"R": self.R.detach().cpu(),
                "atten_logit": None if self.atten_logit is None else self.atten_logit.detach().cpu(),
                "gain": {k: v.detach().cpu() for k, v in self.gain.items()},
                "bias": {k: v.detach().cpu() for k, v in self.bias.items()},
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.R.copy_(state["R"])
            if self.atten_logit is not None:
                self.atten_logit.copy_(state["atten_logit"])
            for k, v in state["gain"].items():
                self.gain[k].copy_(v)
            for k, v in state["bias"].items():
                self.bias[k].copy_(v)
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, model_path):
        """Write refl_layer.pt (the input to self-localization) and images for inspection."""
        import torchvision

        with torch.no_grad():
            R = self.R.detach().cpu()
            torch.save(R, os.path.join(model_path, "refl_layer.pt"))
            torchvision.utils.save_image(R.clamp(0, 1), os.path.join(model_path, "refl_layer.png"))
            peak = float(R.abs().max())
            if peak > 1e-6:
                torchvision.utils.save_image(R.abs() / peak, os.path.join(model_path, "refl_layer_abs.png"))
            A = self.attenuation()
            if A is not None:
                A = A.detach().cpu()
                torch.save(A, os.path.join(model_path, "atten_layer.pt"))
                torchvision.utils.save_image(A.clamp(0, 1), os.path.join(model_path, "atten_layer.png"))
            if self.appearance != "none":
                torch.save({k: (self.gain[k].detach().cpu(), self.bias[k].detach().cpu()) for k in self.gain},
                           os.path.join(model_path, "appearance.pt"))
        print(f"[reflection] saved R (mean |R| {float(R.abs().mean()):.4f}) to {model_path}")

import torch
import torch.optim as optim


class SWALookahead(optim.Optimizer):
    def __init__(
        self,
        optimizer,
        swa=True,
        swa_start=50,
        swa_freq=5,
        swa_lr=None,
        lookahead=True,
        la_steps=5,
        la_alpha=0.8,
    ):
        self.optimizer = optimizer
        self.swa = swa
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.lookahead = lookahead
        self.la_steps = la_steps
        self.la_alpha = la_alpha
        self._step_counter = 0

        if self.swa:
            self.swa_state = [
                [
                    {"n_avg": 0, "swa_buffer": torch.zeros_like(p.data)}
                    for p in param_group["params"]
                ]
                for param_group in self.optimizer.param_groups
            ]

        if self.lookahead:
            self._backup_and_reset()
        super(SWALookahead, self).__init__(optimizer.param_groups, optimizer.defaults)

    def _backup_and_reset(self):
        self.backup_params = [
            p.clone().detach()
            for pg in self.optimizer.param_groups
            for p in pg["params"]
        ]

        self.optimizer.zero_grad()

    def _restore(self):
        for p, backup_p in zip(
            (p for pg in self.param_groups for p in pg["params"]), self.backup_params
        ):
            p.data.copy_(backup_p)

    def _lookahead_step(self):
        for p, backup_p in zip(
            (p for pg in self.optimizer.param_groups for p in pg["params"]),
            self.backup_params,
        ):
            backup_p.data.add_(p.data - backup_p.data, alpha=self.la_alpha)
            p.data.copy_(backup_p.data)

    def _swa_step(self):
        for group, group_swa_state in zip(self.param_groups, self.swa_state):
            for p, state in zip(group["params"], group_swa_state):
                state["n_avg"] += 1
                state["swa_buffer"].add_(
                    p.data - state["swa_buffer"], alpha=1.0 / state["n_avg"]
                )
                p.data.copy_(state["swa_buffer"])

    def update_swa_lr(self):
        if self.swa_lr is not None:
            for param_group in self.param_groups:
                param_group["lr"] = self.swa_lr

    def apply_swa(self):
        for group, swa_state in zip(self.param_groups, self.swa_state):
            for p, state in zip(group["params"], swa_state):
                p.data.copy_(state["swa_buffer"])

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_counter += 1

        if self.lookahead:
            if self._step_counter % self.la_steps == 0:
                self._lookahead_step()

        if (
            self.swa
            and self._step_counter > self.swa_start
            and self._step_counter % self.swa_freq == 0
        ):
            self._swa_step()
            if self.swa_lr is not None and self._step_counter == self.swa_start:
                self.update_swa_lr()

        return loss

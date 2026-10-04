"""Adaptive regulariser strength: hold the hnet regulariser at a fixed share of the loss.

With a fixed beta the regulariser was ~2% of the total loss on spaceEnv_thruster
(cl_s5: task 0.094 vs reg 0.0013), so the hypernetwork was barely held to its old
outputs.  Solving   share = beta*R / (L + beta*R)   for beta gives

    beta = share / (1 - share) * L / R

with L the task loss and R the unweighted regulariser.  Both are EMAs, so beta
follows the trend rather than batch noise, and beta is clamped because R starts
near zero at the beginning of every task (theta still equals the targets).
"""


class RegShareBeta:
    def __init__(self, share, beta_init, beta_min=1e-4, beta_max=1e3, decay=0.99):
        assert 0.0 < share < 1.0, "reg_share_target must be in (0, 1)"
        self.share = share
        self.beta_init = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.decay = decay
        self.task_id = None
        self.reset(None)

    def reset(self, task_id):
        self.task_id = task_id
        self.beta = self.beta_init
        self._L = None
        self._R = None

    def update(self, task_id, loss_task, reg_raw):
        """Feed one batch (floats); returns the beta to weight this batch's regulariser."""
        if task_id != self.task_id:
            self.reset(task_id)
        if self._L is None:
            self._L, self._R = loss_task, reg_raw
        else:
            d = self.decay
            self._L = d * self._L + (1 - d) * loss_task
            self._R = d * self._R + (1 - d) * reg_raw
        # A Gaussian NLL task loss can go negative; the ratio is meaningless then,
        # so keep the last beta.
        if self._L > 0 and self._R > 0:
            beta = self.share / (1.0 - self.share) * self._L / self._R
            self.beta = min(max(beta, self.beta_min), self.beta_max)
        return self.beta

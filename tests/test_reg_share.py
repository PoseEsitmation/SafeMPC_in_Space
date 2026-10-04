from hypercrl.tools.reg_share import RegShareBeta


def test_beta_holds_target_share():
    c = RegShareBeta(0.8, beta_init=0.05)
    for _ in range(500):
        beta = c.update(1, loss_task=0.1, reg_raw=0.02)
    assert abs(beta * 0.02 / (0.1 + beta * 0.02) - 0.8) < 1e-6


def test_beta_clamped_and_reset_per_task():
    c = RegShareBeta(0.8, beta_init=0.05, beta_max=10.0)
    assert c.update(1, loss_task=0.1, reg_raw=1e-9) == 10.0
    c.update(2, loss_task=0.1, reg_raw=0.1)       # new task: EMAs start over
    assert abs(c.beta - 4.0) < 1e-6


def test_negative_task_loss_keeps_beta():
    c = RegShareBeta(0.8, beta_init=0.05)
    assert c.update(1, loss_task=-0.3, reg_raw=0.1) == 0.05

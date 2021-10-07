class EpsilonScheduler:
    """Schedule decay of epislon for epsilon-greedy exploration,
    linear decay from eps_max to eps_min over exploration_fraction * max_epoch * steps_per_spoch steps
    in the environment"""

    def __init__(
        self, max_epoch, step_per_epoch, eps_max, eps_min, exploration_fraction
    ):
        self.exploration_steps = max_epoch * step_per_epoch * exploration_fraction
        self.eps_min = eps_min
        self.eps_max = eps_max

    def current_eps(self, epoch, env_step):
        eps = max(self.eps_min, self.eps_max - env_step / self.exploration_steps)
        return eps


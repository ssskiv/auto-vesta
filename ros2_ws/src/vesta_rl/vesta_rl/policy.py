"""Policy network: CNN on the costmap patch + MLP on the path/state vector."""
import numpy as np
import torch
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from . import config as C


class VestaExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_dim=256, vec_dim=128):
        super().__init__(observation_space, features_dim=cnn_dim + vec_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(),    # 64 -> 30
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),   # 30 -> 14
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),   # 14 -> 6
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 1, C.PATCH_SIZE, C.PATCH_SIZE)).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n, cnn_dim), nn.ReLU())
        self.vec = nn.Sequential(
            nn.Linear(C.VEC_SIZE, vec_dim), nn.ReLU(),
            nn.Linear(vec_dim, vec_dim), nn.ReLU(),
        )

    def forward(self, obs):
        return torch.cat([self.cnn_head(self.cnn(obs['costmap'])), self.vec(obs['vec'])], dim=1)


POLICY_KWARGS = dict(
    features_extractor_class=VestaExtractor,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    log_std_init=-0.5,
)


class VestaPolicy(MultiInputActorCriticPolicy):
    """Gaussian policy whose mean is tanh-bounded to (-1, 1).

    With a plain linear mean, PPO can push it far past the action bound (the
    throttle mean reached ~4.6): every sample is then clipped to full speed,
    so slower speeds are never tried and speed penalties cannot be learned.
    Samples are still mean + std * noise, clipped by the env.
    """

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = torch.tanh(self.action_net(latent_pi))
        return self.action_dist.proba_distribution(mean_actions, self.log_std)


class VestaSquashedPolicy(MultiInputActorCriticPolicy):
    """Squashed Gaussian policy: a = tanh(u), u ~ N(z, sigma), with the tanh
    log-det correction in log pi (as in SAC).

    With VestaPolicy (tanh on the mean, noise added after it) the env clips the
    noisy action, so the average executed action differs from the deterministic
    one: the throttle mean sat at -1 ("stop") and the car only moved thanks to the
    noise. Here every sample is inside (-1, 1), nothing is clipped, and the
    deterministic action tanh(z) is what training actually executes on average.
    Same weights as VestaPolicy, so its checkpoints load unchanged.
    """

    Z_MAX = 3.0  # soft bound on the pre-tanh mean: tanh(3) = 0.995

    def _build(self, lr_schedule):
        self.action_dist = SquashedDiagGaussianDistribution(int(np.prod(self.action_space.shape)))
        super()._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi):
        # Unbounded z drifted to 6+ (throttle "always full"): tanh(u) then rounds to
        # exactly 1 in float32, its log-prob explodes and training diverged to NaN.
        z = self.Z_MAX * torch.tanh(self.action_net(latent_pi) / self.Z_MAX)
        return self.action_dist.proba_distribution(z, self.log_std)


class DeterministicActor(nn.Module):
    """Mean action of an SB3 PPO policy, clipped to [-1, 1]. This is what goes to ONNX."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.squash = isinstance(policy, (VestaPolicy, VestaSquashedPolicy))

    def forward(self, costmap, vec):
        p = self.policy
        features = p.pi_features_extractor({'costmap': costmap, 'vec': vec})
        latent = p.mlp_extractor.forward_actor(features)
        mean = p.action_net(latent)
        if isinstance(p, VestaSquashedPolicy):
            mean = p.Z_MAX * torch.tanh(mean / p.Z_MAX)
        if self.squash:
            mean = torch.tanh(mean)
        return torch.clamp(mean, -1.0, 1.0)

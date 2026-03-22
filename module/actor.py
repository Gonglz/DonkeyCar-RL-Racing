"""
module/actor.py
V13 FiLM-conditioned feature extractor for RecurrentPPO.

Architecture
------------
image (C, H, W) float32 [0,1]          -- 6 semantic channels from CanonicalSemanticWrapper
  └─ CNN → LayerNorm → image_feat (128)    channels: raw_Y, blue_prob, yellow_prob,
                                            sobel_edge, vehicle_prob, motion_residual
state (7,) float32                         v_long_norm, yaw_rate_norm, accel_x_norm,
                                            prev_steer_exec, prev_throttle_exec,
                                            steer_core, bias_smooth
  └─ MLP → h (64) ┬─ film_head (zero-init) → [gamma_raw (128), beta_raw (128)]
                   │    gamma = 1.0 + 0.1 * tanh(gamma_raw)   ← identity at t=0
                   │    beta  = 0.1  * tanh(beta_raw)          ← zero at t=0
                   └─ state_feat_head → state_feat (32)

FiLM:   fused = LayerNorm(gamma * image_feat + beta)   (128)

concat(fused, state_feat) → 160-dim → RecurrentPPO built-in LSTM(128) → actor(3)/value
actor outputs: [Δsteer, speed_scale, line_bias] ∈ [-1,1]³ via ActionAdapterWrapper

Notes
-----
- Vector-level FiLM: γ/β modulate the 128-d CNN feature vector, not spatial maps.
  Adjusts channel-wise importance (e.g. rely more on edge vs yellow at high speed).
- film_head is zero-initialized → identity mapping at init → stable early training.
- LayerNorm on both CNN output and fused output prevents scale drift into LSTM.
- AdaptiveAvgPool2d makes CNN resolution-agnostic (96×96 or 128×128 both work).
- Resume: only compatible with V13-FiLM checkpoints. Old V12 (CombinedExtractor)
  checkpoints will throw a shape mismatch on load.
"""

import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FiLMFeatureExtractor(BaseFeaturesExtractor):
    """
    FiLM-conditioned feature extractor for Dict obs with keys "image" and "state".

    Parameters
    ----------
    observation_space : gym.spaces.Dict
        Must contain:
          "image" : Box(C, H, W) float32 [0, 1]   -- 6-channel semantic image
          "state" : Box(D,)      float32            -- 5D sensor state
    image_feat_dim : int
        Output dimension of CNN backbone and FiLM modulation width. Default 128.
    state_feat_dim : int
        Dimension of state feature concatenated with fused output. Default 32.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        image_feat_dim: int = 128,
        state_feat_dim: int = 32,
    ):
        features_dim = image_feat_dim + state_feat_dim  # 160
        super().__init__(observation_space, features_dim=features_dim)

        n_ch = observation_space["image"].shape[0]  # 6
        self._img_dim = image_feat_dim

        # CNN backbone: (B, C, H, W) → image_feat (image_feat_dim)
        # Stride-2 convolutions downsample 128→64→32→16→8, then AdaptiveAvgPool→4×4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32,  3, stride=2, padding=1), nn.ReLU(),  # → H/2
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(),   # → H/4
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),   # → H/8
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),   # → H/16
            nn.AdaptiveAvgPool2d((4, 4)),                              # → 4×4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, image_feat_dim), nn.ReLU(),
        )
        # LayerNorm before FiLM — stabilizes feature scale entering modulation
        self.image_norm = nn.LayerNorm(image_feat_dim)

        state_dim = observation_space["state"].shape[0]  # 5

        # Shared state encoder trunk
        self.state_enc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())

        # FiLM parameters: γ and β, each image_feat_dim wide
        # film_dim == image_feat_dim (not a separate param) — required by element-wise op
        self.film_head = nn.Linear(64, image_feat_dim * 2)
        # Zero-init: γ_raw=0 → γ=1.0, β_raw=0 → β=0.0 → identity mapping at t=0
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

        # State feature branch (concatenated with fused for LSTM input)
        self.state_feat_head = nn.Sequential(
            nn.Linear(64, state_feat_dim), nn.ReLU()
        )
        # LayerNorm after FiLM — prevents scale drift from γ amplification into LSTM
        self.fused_norm = nn.LayerNorm(image_feat_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        # Image branch: CNN + normalize
        image_feat = self.image_norm(self.cnn(obs["image"]))    # (B, 128)

        # State branch: shared encoder
        h = self.state_enc(obs["state"])                         # (B, 64)

        # FiLM parameters
        film_params = self.film_head(h)                          # (B, 256)
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)       # each (B, 128)

        # Stable FiLM modulation (identity at init)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)               # ∈ (0.9, 1.1)
        beta  = 0.1 * torch.tanh(beta_raw)                       # ∈ (-0.1, 0.1)
        fused = self.fused_norm(gamma * image_feat + beta)        # (B, 128)

        # State feature for concat
        state_feat = self.state_feat_head(h)                     # (B, 32)

        return torch.cat([fused, state_feat], dim=-1)            # (B, 160)

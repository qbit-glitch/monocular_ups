"""CSCMRefineNet: Cross-Modal Refinement Network for semantic pseudo-labels.

Refines CAUSE-TR 27-class semantic pseudo-labels using cross-modal
processing of DINOv2 features conditioned on depth geometry.

Supports two block types:
  - "conv": Depthwise-separable Conv2d blocks (fast, stable, local context)
  - "mamba": VisionMamba2/GatedDeltaNet SSM blocks (long-range, slower)

Architecture:
    DINOv2 (768) → SemanticProjection → sem_proj (bridge_dim)
    DINOv2 (768) + depth → DepthFeatureProjection → depth_proj (bridge_dim)
    [sem_proj, depth_proj] → N × CoupledBlock → [sem_refined, _]
    sem_refined → head → refined_logits (27)

References:
    - Coupled Mamba (Li et al., NeurIPS 2024)
    - DFormerv2 (Yin et al., CVPR 2025)
    - FiLM (Perez et al., AAAI 2018)
    - GatedDeltaNet (Yang et al., ICLR 2025)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba2 import VisionMamba2
except ImportError:
    VisionMamba2 = None  # Mamba blocks unavailable; use block_type="conv"


class SemanticProjection(nn.Module):
    """Project DINOv2 features to bridge dimension for semantic stream."""

    def __init__(self, feature_dim: int = 768, bridge_dim: int = 192):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, bridge_dim, 1, bias=False)
        self.norm = nn.GroupNorm(1, bridge_dim)  # instance-norm style
        self.act = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, feature_dim, H, W) → (B, bridge_dim, H, W)"""
        return self.act(self.norm(self.conv(features)))


class DepthFeatureProjection(nn.Module):
    """Project DINOv2 features with depth FiLM conditioning.

    Encodes depth via sinusoidal positional encoding + Sobel gradients,
    then modulates projected DINOv2 features via FiLM (gamma/beta).
    """

    def __init__(
        self,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        depth_freq_bands: int = 6,
    ):
        super().__init__()
        self.bridge_dim = bridge_dim

        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feature_dim, bridge_dim, 1, bias=False),
            nn.GroupNorm(1, bridge_dim),
            nn.GELU(),
        )

        # Depth encoding: sinusoidal + Sobel gradients → FiLM params
        # Input: sin/cos for each freq band + raw depth + grad_x + grad_y
        depth_input_dim = 2 * depth_freq_bands + 3
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_input_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, bridge_dim * 2, 1),  # gamma + beta for FiLM
        )

        # Pre-compute frequency bands
        self.register_buffer(
            "freq_bands",
            torch.tensor([2**i * math.pi for i in range(depth_freq_bands)]),
        )

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, 768, H, W) DINOv2 patch features
            depth: (B, 1, H, W) normalized depth [0, 1]
            depth_grads: (B, 2, H, W) Sobel_x, Sobel_y of depth
        Returns:
            (B, bridge_dim, H, W) depth-conditioned feature projection
        """
        feat_proj = self.feat_proj(features)

        # Sinusoidal depth encoding
        freqs = self.freq_bands  # (F,)
        d_expanded = depth * freqs[None, :, None, None]  # (B, F, H, W)
        depth_enc = torch.cat([
            torch.sin(d_expanded),
            torch.cos(d_expanded),
            depth,
            depth_grads,
        ], dim=1)  # (B, 2F+3, H, W)

        # FiLM conditioning
        film_params = self.depth_encoder(depth_enc)  # (B, 2*bridge_dim, H, W)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.clamp(-2.0, 2.0)
        beta = beta.clamp(-2.0, 2.0)

        return feat_proj * (1.0 + gamma) + beta


class CoupledConvBlock(nn.Module):
    """Coupled dual-chain Conv2d block for cross-modal feature fusion.

    Two depthwise-separable conv streams (semantic and depth-feature) with
    learnable cross-chain gating. Operates natively in (B, C, H, W) format
    — no flatten/unflatten overhead.
    """

    def __init__(
        self,
        d_model: int = 192,
        coupling_strength: float = 0.1,
        **kwargs,  # accept and ignore Mamba-specific args
    ):
        super().__init__()

        # Pre-norm
        self.norm_sem = nn.GroupNorm(1, d_model)
        self.norm_depth = nn.GroupNorm(1, d_model)

        # Cross-chain coupling: 1×1 conv + sigmoid gate
        self.cross_d2s = nn.Conv2d(d_model, d_model, 1)
        self.cross_s2d = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # Semantic stream: depthwise 3×3 + pointwise expand-contract
        self.sem_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1),
        )

        # Depth stream: same architecture
        self.depth_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model * 2, 1),
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1),
        )

    def forward(
        self, sem: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        # Pre-norm
        sem_n = self.norm_sem(sem)
        depth_n = self.norm_depth(depth_feat)

        # Cross-chain modulation
        sem_input = sem_n + self.alpha * torch.sigmoid(self.cross_d2s(depth_n))
        depth_input = depth_n + self.beta * torch.sigmoid(self.cross_s2d(sem_n))

        # Conv processing + residual
        sem_out = sem + self.sem_conv(sem_input)
        depth_out = depth_feat + self.depth_conv(depth_input)

        return sem_out, depth_out


class CoupledMambaBlock(nn.Module):
    """Coupled dual-chain SSM for cross-modal feature fusion.

    Two separate VisionMamba2 chains (semantic and depth-feature) with
    learnable cross-chain state coupling. Each chain's input is augmented
    by a gated projection of the partner chain's features.

    Reference: Coupled Mamba (Li et al., NeurIPS 2024)
    """

    def __init__(
        self,
        d_model: int = 192,
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
        coupling_strength: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Two independent VisionMamba2 streams
        mamba_kwargs = dict(
            d_model=d_model, scan_mode=scan_mode, layer_type=layer_type,
            d_state=d_state, d_conv=d_conv, expand=expand,
            headdim=headdim, chunk_size=chunk_size,
        )
        self.sem_mamba = VisionMamba2(**mamba_kwargs)
        self.depth_mamba = VisionMamba2(**mamba_kwargs)

        # Cross-chain coupling: gated projection from partner → self
        self.cross_depth_to_sem = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.cross_sem_to_depth = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Learnable coupling strength (initialized small for stable training)
        self.alpha = nn.Parameter(torch.tensor(coupling_strength))
        self.beta = nn.Parameter(torch.tensor(coupling_strength))

        # Pre-norm
        self.norm_sem = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

        # Post-SSM FFN
        self.ffn_sem = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_depth = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self, sem: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sem: (B, D, H, W) semantic stream
            depth_feat: (B, D, H, W) depth-feature stream
        Returns:
            (refined_sem, refined_depth_feat) — same shapes
        """
        B, D, H, W = sem.shape

        # Flatten to (B, L, D) for LayerNorm + cross-coupling
        sem_flat = sem.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_flat = depth_feat.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Pre-norm
        sem_normed = self.norm_sem(sem_flat)
        depth_normed = self.norm_depth(depth_flat)

        # Cross-chain modulation
        cross_d2s = self.alpha * self.cross_depth_to_sem(depth_normed)
        cross_s2d = self.beta * self.cross_sem_to_depth(sem_normed)

        sem_input = sem_normed + cross_d2s
        depth_input = depth_normed + cross_s2d

        # Reshape to (B, D, H, W) for VisionMamba2
        sem_input = sem_input.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_input = depth_input.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Clamp inputs to safe range for SSD backward (exp-cumsum overflows
        # with |x| > ~2.0 in the pure PyTorch reference implementation)
        sem_input = sem_input.clamp(-2.0, 2.0)
        depth_input = depth_input.clamp(-2.0, 2.0)

        # Independent SSM processing with coupled inputs
        sem_out = self.sem_mamba(sem_input)
        depth_out = self.depth_mamba(depth_input)

        # Residual + FFN
        sem_out_flat = sem_out.permute(0, 2, 3, 1).reshape(B, H * W, D)
        depth_out_flat = depth_out.permute(0, 2, 3, 1).reshape(B, H * W, D)

        sem_refined = sem_flat + sem_out_flat
        sem_refined = sem_refined + self.ffn_sem(sem_refined)

        depth_refined = depth_flat + depth_out_flat
        depth_refined = depth_refined + self.ffn_depth(depth_refined)

        # Reshape back to (B, D, H, W)
        sem_refined = sem_refined.reshape(B, H, W, D).permute(0, 3, 1, 2)
        depth_refined = depth_refined.reshape(B, H, W, D).permute(0, 3, 1, 2)

        return sem_refined, depth_refined


class CSCMRefineNet(nn.Module):
    """Cross-Modal Refinement Network for semantic pseudo-labels.

    Takes pre-computed DINOv2 features + SPIdepth depth and produces
    refined 27-class semantic logits.

    Args:
        num_classes: number of semantic classes (27 for CAUSE-TR)
        feature_dim: DINOv2 feature dimension (768 for ViT-B/14)
        bridge_dim: internal bridge dimension
        num_blocks: number of coupled blocks
        block_type: "conv" or "mamba"
        layer_type: "mamba2" or "gated_delta_net" (only for block_type="mamba")
        scan_mode: "raster", "bidirectional", or "cross_scan" (only for block_type="mamba")
        coupling_strength: initial alpha/beta for cross-chain coupling
        d_state: SSM state dimension (only for block_type="mamba")
        chunk_size: SSD chunk size (only for block_type="mamba")
    """

    def __init__(
        self,
        num_classes: int = 27,
        feature_dim: int = 768,
        bridge_dim: int = 192,
        num_blocks: int = 4,
        block_type: str = "conv",
        layer_type: str = "gated_delta_net",
        scan_mode: str = "bidirectional",
        coupling_strength: float = 0.1,
        d_state: int = 64,
        chunk_size: int = 32,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gradient_checkpointing = gradient_checkpointing

        self.sem_proj = SemanticProjection(feature_dim, bridge_dim)
        self.depth_feat_proj = DepthFeatureProjection(feature_dim, bridge_dim)

        if block_type == "conv":
            self.blocks = nn.ModuleList([
                CoupledConvBlock(
                    d_model=bridge_dim,
                    coupling_strength=coupling_strength,
                )
                for _ in range(num_blocks)
            ])
        elif block_type == "mamba":
            self.blocks = nn.ModuleList([
                CoupledMambaBlock(
                    d_model=bridge_dim,
                    layer_type=layer_type,
                    scan_mode=scan_mode,
                    coupling_strength=coupling_strength,
                    d_state=d_state,
                    chunk_size=chunk_size,
                )
                for _ in range(num_blocks)
            ])
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        # Output head: bridge_dim → num_classes
        # Small random init so model starts slightly perturbed from CAUSE
        # (zero-init creates an inescapable local minimum at the CAUSE identity)
        self.head = nn.Sequential(
            nn.GroupNorm(1, bridge_dim),
            nn.Conv2d(bridge_dim, num_classes, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        dinov2_features: torch.Tensor,
        depth: torch.Tensor,
        depth_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dinov2_features: (B, 768, H, W) — DINOv2 patch features
            depth: (B, 1, H, W) — normalized depth [0, 1]
            depth_grads: (B, 2, H, W) — Sobel gradients of depth
        Returns:
            refined_logits: (B, 27, H, W)
        """
        sem = self.sem_proj(dinov2_features)
        depth_feat = self.depth_feat_proj(dinov2_features, depth, depth_grads)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                sem, depth_feat = checkpoint(
                    block, sem, depth_feat, use_reentrant=False,
                )
            else:
                sem, depth_feat = block(sem, depth_feat)

        refined_logits = self.head(sem)

        return refined_logits

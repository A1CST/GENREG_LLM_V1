# ================================================================
# GENREG Evolved Encoder — GPU Tensor Operations
# ================================================================
# GPU-batched version of the encoder for use in GPUEvolver v3.
#
# Each genome has:
#   - Encoder weights: enc_w (B, enc_dim, input_dim), enc_b (B, enc_dim)
#   - Activation ID: act_ids (B,) — index into catalog
#   - Activation params: act_p1..act_p4 (B, enc_dim) — up to 4 params
#
# All activations are implemented as differentiable tensor ops
# so the entire forward pass stays on GPU.
# ================================================================

import torch
import torch.nn.functional as F

NUM_ACTIVATIONS = 8

# ================================================================
# GPU ACTIVATION FUNCTIONS (vectorized, batched)
# ================================================================
# All take: x (B, D), p1-p4 (B, D) and return (B, D)
# Unused params are ignored per activation.
#
# Mapping to CPU catalog:
#   0: tanh_scaled     — p1=alpha, p2=beta, p3=gamma
#   1: gated_linear    — p1=gate, p2=scale
#   2: soft_threshold  — p1=sharpness, p2=threshold, p3=scale
#   3: resonance       — p1=amp, p2=freq, p3=phase
#   4: dual_path       — p1=w_tanh, p2=s_tanh, p3=w_lin, p4=s_lin
#   5: abs_gate        — p1=scale, p2=rate
#   6: quadratic_relu  — p1=threshold, p2=scale
#   7: identity_plus   — p1=nudge, p2=bend
# ================================================================

def _gpu_tanh_scaled(x, p1, p2, p3, p4):
    return p1 * torch.tanh(p2 * x + p3)

def _gpu_gated_linear(x, p1, p2, p3, p4):
    return p2 * x * torch.sigmoid(p1 * x)

def _gpu_soft_threshold(x, p1, p2, p3, p4):
    return p3 * torch.sigmoid(p1 * (x - p2))

def _gpu_resonance(x, p1, p2, p3, p4):
    return p1 * torch.sin(p2 * x + p3)

def _gpu_dual_path(x, p1, p2, p3, p4):
    return p1 * torch.tanh(p2 * x) + p3 * x * torch.sigmoid(p4 * x)

def _gpu_abs_gate(x, p1, p2, p3, p4):
    return p1 * (1.0 - torch.exp(-p2 * x.abs()))

def _gpu_quadratic_relu(x, p1, p2, p3, p4):
    v = F.relu(x - p1)
    return p2 * v * v

def _gpu_identity_plus(x, p1, p2, p3, p4):
    return x + p1 * torch.tanh(p2 * x)

GPU_ACTIVATIONS = [
    _gpu_tanh_scaled,
    _gpu_gated_linear,
    _gpu_soft_threshold,
    _gpu_resonance,
    _gpu_dual_path,
    _gpu_abs_gate,
    _gpu_quadratic_relu,
    _gpu_identity_plus,
]


def apply_evolved_activations(x, act_ids, p1, p2, p3, p4):
    """
    Apply per-genome activation functions.

    x:       (B, D) — pre-activation encoder output
    act_ids: (B,)   — activation index per genome
    p1..p4:  (B, D) — activation params per genome per dim

    Returns: (B, D) activated output

    Strategy: compute ALL activations for all genomes, then select
    the right one per genome via gather. This avoids branching and
    keeps everything as tensor ops on GPU.
    """
    B, D = x.shape

    # Stack all 8 activation outputs: (8, B, D)
    all_outputs = torch.stack([fn(x, p1, p2, p3, p4) for fn in GPU_ACTIVATIONS])

    # Select per-genome: act_ids (B,) → index into dim 0
    idx = act_ids.view(1, B, 1).expand(1, B, D)  # (1, B, D)
    selected = all_outputs.gather(0, idx).squeeze(0)  # (B, D)

    return selected


# ================================================================
# DEFAULT PARAMS PER ACTIVATION (for initialization)
# ================================================================
# (p1_default, p2_default, p3_default, p4_default)
DEFAULT_PARAMS = [
    (1.0, 1.0, 0.0, 0.0),   # tanh_scaled: alpha, beta, gamma, _
    (1.0, 1.0, 0.0, 0.0),   # gated_linear: gate, scale, _, _
    (3.0, 0.5, 1.0, 0.0),   # soft_threshold: sharpness, threshold, scale, _
    (1.0, 2.0, 0.0, 0.0),   # resonance: amp, freq, phase, _
    (0.5, 1.0, 0.5, 1.0),   # dual_path: w_tanh, s_tanh, w_lin, s_lin
    (1.0, 2.0, 0.0, 0.0),   # abs_gate: scale, rate, _, _
    (0.0, 1.0, 0.0, 0.0),   # quadratic_relu: threshold, scale, _, _
    (0.3, 1.0, 0.0, 0.0),   # identity_plus: nudge, bend, _, _
]

# Bounds for mutation clamping: (p1_lo, p1_hi, p2_lo, p2_hi, p3_lo, p3_hi, p4_lo, p4_hi)
PARAM_BOUNDS = [
    (-3, 3,   0.1, 5,   -2, 2,    -1, 1),
    (0.1, 5,  -3, 3,    -1, 1,    -1, 1),
    (0.5, 10, -1, 2,    -3, 3,    -1, 1),
    (-2, 2,   0.5, 8,   -3.14, 3.14, -1, 1),
    (-2, 2,   0.1, 5,   -2, 2,    0.1, 5),
    (-3, 3,   0.1, 8,   -1, 1,    -1, 1),
    (-1, 1,   0.1, 5,   -1, 1,    -1, 1),
    (-1, 1,   0.1, 5,   -1, 1,    -1, 1),
]

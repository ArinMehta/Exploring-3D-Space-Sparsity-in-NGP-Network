# Exploring-3D-Space-Sparsity-in-NGP-Network

This project investigates a lightweight extension to the Instant-NGP framework by introducing a **Trainable Additive Saliency Grid** for improved 3D geometry representation and mesh extraction.

## Overview

Extracting accurate and clean 3D meshes from radiance fields like NeRF is challenging. Our approach modifies the **Instant-NGP** architecture by introducing a **trainable 3D grid** that adds a spatial bias to the raw log-density values. This helps guide density learning more effectively, boosting mesh quality and interpretability.

### Key Contributions

- **Trainable Additive Grid (`G`)** modifies density pre-activation outputs using spatial priors.
- Retains **Instant-NGP acceleration** and architecture (hash grid, compact MLP, bitfield-based ray marching).
- Demonstrates improved **density saliency maps** and mesh quality with minimal architectural overhead.

## Method

1. **Base Architecture**:
   - Built on [`ngp_pl`](https://github.com/kwea123/ngp_pl), a PyTorch Lightning implementation of Instant-NGP.
   - Uses 3D hash grid encoding (`L=16, F=1, T=219`) and Spherical Harmonics for direction encoding.
   - MLP outputs: `σ_raw` (log-density) + `f_geo` (15D geometric features).

2. **Additive Saliency Grid**:
   - 3D learnable grid `G` (e.g., 64³) initialized to zero.
   - Grid interpolates bias `g` at each location `x`, added to raw density:
     σ_combined = σ_raw + α * g      # α ≈ 0.1 <br/>
     σ = exp( clamp(σ_combined, max=15) )
   - Enables direct learning of sparse, spatial priors.

3. **Color Prediction**:
   - Handled by a separate MLP using view direction + geometry features to predict RGB.

4. **Loss Function**:
   - Combined loss: `L = L_rgb (MSE) + λ_s * L_saliency`

5. **Meshing**:
   - Post-training Marching Cubes on the final density field (256³ resolution) using tuned threshold `τ`.

## Results

- Achieves **PSNR comparable to baseline Instant-NGP (F=2)** while using F=1 and an additive grid.
- **Clearer saliency maps** with better separation of surfaces vs empty space.
- Extracted meshes (e.g., Chair, Lego) are qualitatively cleaner and more complete.

## Future Work

- **Feature Vector Scaling**: Multiply full feature vector by saliency weights.
- **Zero-Skipping Gates**: Use `tanh`-based gates to enforce near-zero density in low-saliency regions.
- **ADMM Pruning**: Replace L1 regularization with ADMM-based sparsity enforcement (as in HollowNeRF).

## References

1. Müller et al., *Instant Neural Graphics Primitives*, [ACM TOG 2022](https://doi.org/10.1145/3528223.3530127)
2. Xie et al., *HollowNeRF*, [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_HollowNeRF_Pruning_Hashgrid-Based_NeRFs_with_Trainable_Collision_Mitigation_ICCV_2023_paper.pdf)

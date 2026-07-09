# D²R²OSR: Degradation-Disentangled Representation for Real-World Omnidirectional Image Super-Resolution

Official inference code for the ECCV 2026 paper
**"D²R²OSR: Degradation-Disentangled Representation for Real-World Omnidirectional Image Super-Resolution"**.

> Hongyu An, Xinfeng Zhang, Xu Fan, Shijie Zhao, Li Zhang, Ruiqin Xiong.
> *University of Chinese Academy of Sciences · ByteDance Inc. · Peking University.*

Omnidirectional images (ODIs) suffer from a mix of real-world degradations
(blur, noise, resize, compression) introduced during fisheye capture, plus
geometric distortions introduced by Equirectangular Projection (ERP). D²R²OSR
disentangles the two with a dual-branch architecture:

- **PPR** — Perspective Projection Representation: a continuous, near-lossless
  ERP↔perspective coordinate mapping (via a learned Jacobian/Hessian-based
  estimator) that gives the network a viewpoint-centric, projection-independent
  view alongside the ERP view.
- **DSM** — Degradation-Specific Module, embedded in both the **EAMB**
  (ERP-Adaptive Modulation Block, models latitude-dependent geometric
  distortion) and **PAMB** (PPR-Adaptive Modulation Block, models real-world
  pixel-level degradation).
- **PFAM** — Projection Fusion Attention Module: adaptive spatial-channel
  gated fusion that combines the ERP and PPR branches at every stage.

The codebase is built on [BasicSR](https://github.com/XPixelGroup/BasicSR)
(the project was originally forked from
[OSRT](https://github.com/Fanghua-Yu/OSRT), CVPR 2023).

## Status of this release

> ⚠️ **Inference only, ×4 only, for now.** This repository currently ships the
> minimal code path needed to run the pretrained D²R²OSR model on ODIs at
> ×4. The paper also reports ×8/×16 results; those configs/checkpoints are
> not part of this release yet. Training code, the full degradation-synthesis
> pipeline, and the baseline/comparison methods used in the paper's tables
> are not included in this drop either.

## Code ↔ paper name mapping

The paper's method name (D²R²OSR) and module names (EAMB/PAMB/DSM/PFAM) were
finalized after the code was written. The implementation originally used an
internal experiment codename (`DROSR_PPR_Fusion_Prompt3`, one of ~20 fusion/
prompt variants tried during development); this release renames the exact
final model — confirmed by cross-checking `network_g.type` against the
×4/×8/×16 test configs and the paper's per-scale results table (this repo
ships the ×4 config; see [Status of this release](#status-of-this-release)) —
to match the paper:

| Paper | Code |
| --- | --- |
| D²R²OSR (full model) | `D2R2OSR` in [`d2r2osr/archs/d2r2osr_arch.py`](d2r2osr/archs/d2r2osr_arch.py) |
| PPR (ERP↔perspective mapping) | [`d2r2osr/utils/e2p.py`](d2r2osr/utils/e2p.py) (`e2p_patch`) |
| EAMB / PAMB + DSM | `RHAG_Prompt` / prompt-conditioned blocks from [`hat_arch.py`](d2r2osr/archs/hat_arch.py), [`swinir_arch.py`](d2r2osr/archs/swinir_arch.py), [`prompt_arch.py`](d2r2osr/archs/prompt_arch.py) |
| PFAM | fusion modules (`ChannelAttentionFusion` / `SpatialCrossAttentionFusion` / `AdaptiveFusion`) in the arch file above |
| Test-time runner | `D2R2OSRModel` in [`d2r2osr/models/d2r2osr_model.py`](d2r2osr/models/d2r2osr_model.py) |

## Installation

```bash
pip install -r requirements.txt
```

This needs the `basicsr` package (standard `pip install basicsr`, or install
from source if you need a newer/patched version) — `test.py` is a thin
BasicSR entry point, and all archs/models/datasets register themselves into
BasicSR's registry on import.

## Inference

1. **Get the pretrained checkpoints** (not tracked in this repo — see
   [Checkpoints](#checkpoints) below) and place them under `pretrained/` as
   referenced by the configs, or edit `path.pretrain_network_g` in the yml.

2. **Point the config at your test data.** Edit `datasets.test_*.dataroot_gt`
   / `dataroot_lq` in the relevant config to your ODI test set (ERP images,
   GT at 1024×2048). `condition_type: cos_latitude` is the latitude-prior
   channel consumed by the EAMB/DSM — no extra preprocessing is needed, it is
   computed by the dataset loader.

3. **Run:**

   ```bash
   python test.py -opt options/test/d2r2osr_x4.yml
   ```

   The config has a `tile` section (`tile_size` / `tile_pad`) to bound GPU
   memory on high-resolution ERP inputs — lower `tile_size` if you run out of
   memory, at a small cost to boundary consistency.

## Checkpoints

Not tracked in this repo (~400 MB). Download `d2r2osr_x4.pth` and place it
under `pretrained/` as referenced by the config, or edit
`path.pretrain_network_g`.

> **Download:** [Google Drive](https://drive.google.com/file/d/1LRY-yt3Sr52Bp5dlRmuxZXNTbpiqhGPN/view?usp=sharing)

## Datasets

The paper trains on **Flickr360** and evaluates on **Flickr360**, **ODI-SR**,
and **SUN360** — all ERP ODIs at 1024×2048.

- Flickr360: released with the
  [NTIRE 2023 360° Omnidirectional Image and Video Super-Resolution challenge](https://github.com/360SR/360SR-Challenge).
- ODI-SR: from [LAU-Net](https://github.com/Panorama-3D/LAU-Net) (Deng et al.,
  CVPR 2021).
- SUN360: [Xiao et al., CVPR 2012](https://vision.princeton.edu/projects/2012/SUN360/).

**TODO — add the exact train/test split files used in the paper.**

For inference you only need HR (or already-degraded LR) ERP images at
1024×2048 per the test configs' `gt_h`/`gt_w`.

### Preparing training data (for retraining, not needed for inference)

1. Crop HR ERP images into training sub-images:

   ```bash
   python scripts/extract_subimage.py \
       --input-folder datasets/train/Flickr360/HR \
       --save-folder datasets/train/Flickr360/HR_sub \
       --wh 2048 1024 --crop-size 512 --step 256
   ```

2. Generate the BasicSR meta_info file consumed by `dataroot_gt` /
   `meta_info_file` in the train configs:

   ```bash
   python scripts/generate_meta_info.py \
       --gt-folder datasets/train/Flickr360/HR_sub \
       --meta-info-txt datasets/train/Flickr360/meta_info_HR_sub.txt
   ```

3. LQ images are **not** pre-generated — [`RealESRGANODISRDataset`](d2r2osr/data/realesrgan_odisr_dataset.py)
   synthesizes the mixed fisheye→ERP real-world degradation on the fly during
   training (Real-ESRGAN-style second-order degradation, extended with the
   fisheye/ERP-specific distributions from the paper). To generate a fixed LR
   set for evaluation instead (matching the paper's fisheye degradation path),
   use:

   ```bash
   python scripts/erp_downsample.py --input HR.png --output LR.png --scale 4
   ```

## Acknowledgements

This work is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR) and
was originally forked from [OSRT](https://github.com/Fanghua-Yu/OSRT). It also
reuses architectural components from
[HAT](https://github.com/XPixelGroup/HAT) and
[SwinIR](https://github.com/JingyunLiang/SwinIR). Please follow and star
those repositories.

## Citation

```bibtex
@inproceedings{an2026d2r2osr,
  title     = {D{\textasciicircum}2R{\textasciicircum}2OSR: Degradation-Disentangled Representation for Real-World Omnidirectional Image Super-Resolution},
  author    = {An, Hongyu and Zhang, Xinfeng and Fan, Xu and Zhao, Shijie and Zhang, Li and Xiong, Ruiqin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

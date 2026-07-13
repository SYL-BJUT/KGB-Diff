# KGB-Diff

> Knowledge Graph–Guided Cross-Modal Generation for Incomplete Palmprint and Palm Vein Fusion Recognition

This repository hosts the official implementation of KGB-Diff, a diffusion-based framework that recovers missing biometric modalities (palmprint or palm vein) from the available one, conditioned on a structured **Knowledge Graph (KG)** of cross-identity cross-modal relationships. The recovered modality is then fused with the available modality for identification.

This repo currently hosts the **baseline IP-Adapter implementation and the eval pipeline**. The full KG-enhanced training and inference code will be released upon paper acceptance.

---

## Why KGB-Diff

Existing IP-Adapter–style cross-modal generation trains **one conditional per identity**: it maps a single palmprint (or palm vein) to its vein (or print). When only one modality is captured at inference time (the *incomplete acquisition* scenario), the model has to hallucinate the missing one from a single query, which is fundamentally under-constrained.

KGB-Diff instead conditions generation on **structured knowledge aggregated from many identities** through a knowledge graph:

- nodes: identity, palmprint sample, palm vein sample, palmprint CCNet feature, palm vein CCNet feature
- edges: same-identity links (print ↔ vein), same-identity print ↔ print, and cross-identity visual-similarity edges discovered via CCNet feature retrieval

At training time, KG-paired samples give the diffusion model **contextual conditioning** (i.e., it learns from many *similar* identities at once). At inference time, the KG graph retrieves the most relevant neighbors for the query, and the generator synthesizes the missing modality using both the query and the retrieved KG context.

---

## Quick Start (baseline only)

### 1. Install

```bash
# Linux / Windows (PowerShell). Python >= 3.9.
python -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```



### 2. Get the pretrained models

| Component | What it is |
|-----------|------------|
| **Stable Diffusion v1.5** (UNet + VAE + scheduler) | public, downloaded from HuggingFace |
| **CLIP-ViT-L/14 image encoder** | public, downloaded from HuggingFace |
| **IP-Adapter modules** (the open-source `IP-Adapter-main`) | we vendor it as a submodule / upstream clone |
| **CCNet weights** (palmprint & palm vein) | ours, released here |
| **Baseline IP-Adapter checkpoints** (P2V and V2P) | ours, released here |

### 3. Prepare data

Download the datasets and place them under the conventional layout expected by our preprocessing scripts:

```
data/
├── Tongji600/
│   ├── PalmPrint/      ← PalmPrint_<id>_<sample>.png
│   └── PalmVein/       ← PalmVein_<id>_<sample>.png
├── PolyU-MB-V2/
│   ├── PalmPrint/      ← palm_<NNNNNN>.png
│   └── PalmVein/       ← palm_<NNNNNN>.png
└── CUMT/
    ├── PalmPrint/      ← <id>_<sample>.png   (nested structure supported)
    └── PalmVein/       ← <id>_<sample>.png
```

Then preprocess and extract CCNet features:

```bash
python train.py ^
  --batch_size 16 ^
  --epoch_num 1500 ^
  --lr 0.0001 ^
  --id_num 600 ^
  --com_weight 0.8 ^
  --weight1 0.8 ^
  --weight2 0.2 ^
  --temp 0.07 ^
  --redstep 100 ^
  --test_interval 50 ^
  --save_interval 50 ^
  --train_set_file "<PATH_TO_TRAIN_TXT>" ^
  --test_set_file "<PATH_TO_TEST_TXT>" ^
  --des_path "<PATH_TO_RESULTS_DIR>" ^
  --path_rst "<PATH_TO_RESULTS_DIR>" ^
  --gpu_id 0
```

### 4. Run baseline training

Palmprint → palm vein:

```bash
python train_P2V_gen.py ^
  --pretrained_model_name_or_path "<PATH_TO_STABLE_DIFFUSION_V1_5>" ^
  --image_encoder_path          "<PATH_TO_CLIP_VISION_ENCODER>" ^
  --palmprint_dir               "<PATH_TO_TRAIN_PALMPRINT_DIR>" ^
  --palmvein_dir                "<PATH_TO_TRAIN_PALMVEIN_DIR>" ^
  --output_dir                  "<PATH_TO_P2V_OUTPUT_DIR>" ^
  --resolution                  128 ^
  --train_batch_size            64 ^
  --num_train_epochs            100 ^
  --learning_rate               1e-4 ^
  --save_steps                  500 ^
  --mixed_precision             fp16
```

Palm vein → palmprint (128×128 resolution, matches the V2P protocol):

```bash
python train_V2P_gen.py ^
  --pretrained_model_name_or_path "<PATH_TO_STABLE_DIFFUSION_V1_5>" ^
  --image_encoder_path          "<PATH_TO_CLIP_VISION_ENCODER>" ^
  --palmvein_dir                "<PATH_TO_TRAIN_PALMVEIN_DIR>" ^
  --palmprint_dir               "<PATH_TO_TRAIN_PALMPRINT_DIR>" ^
  --output_dir                  "<PATH_TO_V2P_OUTPUT_DIR>" ^
  --resolution                  128 ^
  --train_batch_size            64 ^
  --num_train_epochs            100 ^
  --learning_rate               1e-4 ^
  --save_steps                  500 ^
  --mixed_precision             fp16
```

These scripts produce a generated ROI for each query, mirroring the eval protocol. No KG is used at any step; this is the IP-Adapter baseline as reported in Table X of the paper.

### 4. Run baseline inference

Palmprint → palm vein:

```bash
python inference_baseline_palm2vein.py ^
  --sd_model_path       "<PATH_TO_STABLE_DIFFUSION_V1_5>" ^
  --image_encoder_path  "<PATH_TO_CLIP_VISION_ENCODER>" ^
  --ip_adapter_path     "<PATH_TO_P2V_BASELINE_IP_ADAPTER_BIN>" ^
  --input_dir           "<PATH_TO_TEST_PALMPRINT_DIR>" ^
  --output_dir          "<PATH_TO_GENERATED_VEINS_OUTPUT_DIR>" ^
  --resolution          128 ^
  --num_inference_steps 50 ^
  --seed                42
```

Palm vein → palmprint (128×128 resolution, matches the V2P protocol):

```bash
python inference_baseline_vein2palm_128.py ^
  --sd_model_path       "<PATH_TO_STABLE_DIFFUSION_V1_5>" ^
  --image_encoder_path  "<PATH_TO_CLIP_VISION_ENCODER>" ^
  --ip_adapter_path     "<PATH_TO_V2P_BASELINE_IP_ADAPTER_BIN>" ^
  --input_dir           "<PATH_TO_TEST_PALMVEIN_DIR>" ^
  --output_dir          "<PATH_TO_GENERATED_PALMPRINTS_OUTPUT_DIR>" ^
  --resolution          128 ^
  --num_inference_steps 50 ^
  --seed                42
```

---

## Code origin & acknowledgements

- **IP-Adapter** module is vendored from the official `tencent-ailab/IP-Adapter` release. 
- **CCNet** (palmprint/vein recognition baseline) reuses the architecture and weights from the printed CCNet paper. 
- **Stable Diffusion v1.5** and **CLIP-ViT-L/14** are downloaded from public HuggingFace repositories; the licenses of those originals apply.

We thank the authors of the above works for releasing their code.

---

## Datasets

| Dataset | Public | Used for |
|---------|--------|----------|
| Tongji | yes, contact authors of Tongji Univ. | P2V, V2P, fusion |
| PolyU Multi-spectral palmprint V2 | yes | P2V, V2P, fusion |
| CUMT palmprint | yes | P2V, V2P, fusion |

If you use any of these datasets, please cite the original sources.

---




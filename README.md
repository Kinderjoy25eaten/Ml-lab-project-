# Underwater Acoustic Target Recognition (Feature Fusion + NCA + ResNet18)

Implementation of the paper **"Underwater Acoustic Target Recognition Method Based on Feature Fusion and Residual CNN"** (Yang, Yao, Wang, *IEEE Sensors Journal*, vol. 24, no. 22, 2024, DOI: [10.1109/JSEN.2024.3464754](https://doi.org/10.1109/JSEN.2024.3464754)).

**Authors:** Rup Narayan Jha, Jaypal Malviya

## What this project does

Classifies a ship as **Tug, Cargo, Tanker or Passenger ship** from its underwater sound, using the [DeepShip](https://github.com/irfankamboh/DeepShip) dataset.

## Method

1. **Preprocess** audio: pre-emphasis, 256-sample frames (hop 128), Hamming window
2. **Extract 4 features** per frame: MFCC (12), GFCC (31), CQT (41), LOFAR (50)
3. **Fuse** them into MGCL (134), then add Delta features to get MGCL-Delta (268)
4. **Select features** with NCA (Neighborhood Component Analysis, threshold 0.3), 268 → 83
5. **Build images** from 5 consecutive frames (3 × 5 × 83)
6. **Classify** with ResNet18 (max-pool removed, 17 conv layers + 1 FC)

## Results

| | Accuracy |
|---|---|
| Paper | 97.29% |
| Our run (random 3:1 split) | 99.62% |

- Precision, recall and F1 are all about 99.6%.
- NCA kept 83 of 268 features (the paper kept 96).
- **Note:** the random split puts clips from the same recording in both train and test, so our number is likely optimistic. A recording-level split (`SPLIT_MODE = "recording"` in the notebook) gives a stricter measure. The paper's settings are not fully specified, so some choices are our own and are marked `[CHOICE]` in the notebook.

## Files

| File | Description |
|---|---|
| `uatr_mgcl_full_notebook.ipynb` | Full implementation, runs top to bottom |
| `UATR_presentation.pptx` | 5-slide project presentation |

The paper itself is not included because of IEEE copyright. Please use the DOI link above.

## How to run

```bash
pip install jupyter numpy scipy pandas matplotlib scikit-learn soundfile torch torchvision
jupyter notebook uatr_mgcl_full_notebook.ipynb
```

The notebook downloads DeepShip from GitHub on its own. A GPU is recommended for training.

## Reference

Y. Yang, Q. Yao and Y. Wang, "Underwater Acoustic Target Recognition Method Based on Feature Fusion and Residual CNN," *IEEE Sensors Journal*, vol. 24, no. 22, pp. 37342-37357, 2024.

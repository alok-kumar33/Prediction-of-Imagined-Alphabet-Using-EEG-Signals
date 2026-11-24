# Prediction-of-Imagined-Alphabet-Using-EEG-Signals
# Decoding Imagined Letters from EEG

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-neural%20networks-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

Five-class imagined alphabet decoding (**A, K, N, P, Z**) from OpenBCI EEG recordings. The project compares a band-power SVM baseline with two small neural networks: an EEGNet-like model and **TA-MEEGNet** (a multi-scale EEGNet with attention pooling), under two testing protocols: same person and new person.

**Authors:** Alok Kumar Sah, Manish Kumar Sah, Deependra Kumar Gupta (NIT Warangal)

## Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Results](#results)
5. [Repository structure](#repository-structure)
6. [Getting started](#getting-started)
7. [Output files](#output-files)
8. [Limitations](#limitations)
9. [Future work](#future-work)
10. [Citation and acknowledgements](#citation-and-acknowledgements)

## Overview

People wore an EEG headset and imagined one of five letters. We train models to guess the letter from the recording and test them in two ways:

| Protocol | Meaning |
| --- | --- |
| **Within-subject** | Train and test on the same person |
| **LOSO** (leave one subject out) | Train on other people, test on a person the model has never seen |

With 5 balanced classes, random guessing gets **20%**. All results should be read against this chance level.

## Dataset

*An EEG Dataset for Five-Class Imagined Alphabet Decoding Using OpenBCI Recordings*, Mendeley Data, DOI [10.17632/cp586sfzrt.4](https://data.mendeley.com/datasets/cp586sfzrt) (Version 4).

| Property | Value |
| --- | --- |
| People | 20 |
| Classes | 5 (A, K, N, P, Z) |
| Trials | 25 per letter per person, 2,500 in total |
| Channels | 15 |
| Sampling rate | 125 Hz |
| Storage | 390,000 windows = 2,500 trials x 156 windows x 8 readings |

The data is **not included** in this repository. Download it from Mendeley and check the dataset's licence terms there before redistributing anything.

## Method

1. Load the preprocessed MATLAB (v7.3 / HDF5) epoch file.
2. Rebuild 2,500 trials of shape 15 x 1,248 and verify trial order, labels and participants.
3. Scale each trial and channel on its own (an optional 1-40 Hz band-pass is off by default).
4. Split by **whole trials**, never by windows. The code stops with an error if train, validation and test sets overlap.
5. Train three models:

| Model | Description | Parameters |
| --- | --- | --- |
| BandPower+SVM | Log power in 5 bands (delta, theta, alpha, beta, gamma) on 15 channels = 75 features, linear SVM | n/a |
| EEGNet-like | One temporal filter length (63), average pooling over time | 1,773 |
| TA-MEEGNet | Three temporal filter lengths (15, 31, 63) plus attention pooling | 4,254 |

6. Evaluate with accuracy, balanced accuracy, macro-F1, bootstrap confidence intervals and a paired Wilcoxon test.
7. Extra experiments: ablations (single-scale, or no attention) and few-shot calibration (0, 1 or 5 labelled trials per letter from the new person).

Split sizes: within-subject 75 train / 25 validation / 25 test trials per person. LOSO 2,000 train (16 people), 375 validation (3 people), 125 test (1 person).

Training settings (networks): AdamW, learning rate 0.001, weight decay 0.01, batch size 64, dropout 0.5, up to 60 epochs with early stopping (patience 12), small time-shift and noise augmentation on training data only, seed 0.

## Results

Mean over 20 people. Chance is 0.20.

### Within-subject

| Model | Accuracy |
| --- | --- |
| BandPower+SVM | 0.856 |
| TA-MEEGNet | 0.850 |
| EEGNet-like | 0.812 |

### LOSO (new person)

| Model | Balanced accuracy | 95% bootstrap CI |
| --- | --- | --- |
| EEGNet-like | 0.234 | 0.202 to 0.267 |
| SingleScale+Attn | 0.210 | 0.180 to 0.239 |
| TA-MEEGNet | 0.205 | 0.176 to 0.238 |
| MultiScale+Avg | 0.198 | 0.163 to 0.238 |
| BandPower+SVM | 0.194 | 0.165 to 0.223 |

TA-MEEGNet vs EEGNet-like in LOSO: Wilcoxon p = 0.212 (not significant).

### Few-shot calibration (TA-MEEGNet, LOSO)

| Labelled trials per letter | Accuracy |
| --- | --- |
| 0 | 0.206 |
| 1 | 0.236 |
| 5 | 0.250 |

### Figures

![LOSO score per person](figures/loso_per_person.png)

![Confusion matrix, TA-MEEGNet LOSO](figures/confusion_TA-MEEGNet_loso.png)

### Summary

- Within-subject decoding works (about 81-86%).
- No model clearly decodes a new person (about 19-23%, close to chance).
- Multi-scale filters and attention gave no clear benefit in LOSO.
- A few labelled trials from the new person help a little, but stay far below within-subject accuracy.

## Repository structure

```
.
├── README.md
├── notebooks/
│   └── EEG_Alphabet_Project.ipynb     # the Colab notebook (Cells 1-15)
├── results/
│   ├── summary.csv
│   ├── per_person_results.csv
│   └── calibration_results.csv
├── figures/
│   ├── loso_per_person.png
│   └── confusion_<model>_<protocol>.png
└── docs/
    └── project_report.docx            # optional
```

## Getting started

No local installation is needed. The project runs in Google Colab, which already includes PyTorch, scikit-learn, SciPy, h5py, pandas, matplotlib and seaborn.

1. Download the dataset from the Mendeley link above and upload the `.mat` files to a Google Drive folder named `EEG_project`.
2. Open [Google Colab](https://colab.research.google.com), open `notebooks/EEG_Alphabet_Project.ipynb`, and choose **Runtime > Change runtime type > GPU**.
3. Run the cells in order:

| Cell | Purpose |
| --- | --- |
| 1 | Check Python, PyTorch and the GPU |
| 2 | Mount Drive, set `SRC`, `DATA`, `OUT` |
| 3 | List `.mat` and `.csv` files |
| 4 | Print variable names and shapes inside the `.mat` file |
| 5 | Rebuild trials and run safety checks |
| 6 | Save `trials.npz` to Drive (optional) |
| 7 | Preprocessing |
| 8 | Splits, leakage checks, metrics |
| 9 | BandPower+SVM baseline |
| 10 | Model and training code |
| 11 | Within-subject run |
| 12 | LOSO run (main result) |
| 13 | Few-shot calibration |
| 14 | Ablations |
| 15 | Tables, confidence intervals, statistics, plots |

Run a quick test with `PEOPLE = [0, 1, 2]` first, then switch to `PEOPLE = ALL_SUBJECTS`. The LOSO run is the slowest step. If Colab disconnects, re-run from Cell 1 (or load `trials.npz` to skip Cells 3-5).

### Expected variables in the `.mat` file

`X` (8 x 15 x 390000 in HDF5 order), `y`, `labels`, `participants`, `trial_uid`, `window_uid`, `class_names`, `channel_names`, `fs`, `preprocessing`, `feature_mode`. If your copy differs, adjust Cell 5.

## Output files

| File | Contents |
| --- | --- |
| `trials.npz` | Rebuilt trial array, labels and person IDs |
| `summary.csv` | One row per model and protocol, with means and confidence intervals |
| `per_person_results.csv` | Score of each model for each person |
| `calibration_results.csv` | Few-shot calibration results |
| `confusion_<model>_<protocol>.png` | Normalised confusion matrices |
| `loso_per_person.png` | LOSO score of each held-out person, with chance line |

## Limitations

- Each result comes from a single training run (seed 0). Other seeds may shift the numbers.
- Within-subject splits are random. If trials of the same letter were recorded close together in time, these scores could be optimistic. Recording order was not checked.
- In the within-subject test, each person has only 25 test trials, so one trial is worth 4 percentage points.
- Three-person quick runs are for debugging only and are not reported as results.

## Future work

- Repeat with several seeds and report the spread.
- Add a time-ordered split (train on earlier trials, test on later ones).
- Try alignment or domain-adaptation methods for cross-person transfer.
- Run the dataset authors' own baselines for comparison.


## License

<!-- Add a LICENSE file to the repository and state the licence here. -->

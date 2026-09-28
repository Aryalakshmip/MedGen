# MedGen: Explainable Multi-View Radiology Report Generation with Relational Memory Transformers

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-BioBERT%20%7C%20ClinicalBERT-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)]()

> **MedGen** is an end-to-end multimodal deep learning framework designed to automatically generate structured, clinically accurate diagnostic radiology reports from multi-view chest radiographs (frontal and lateral). It integrates **Deformable & Semantic Dilated Convolutions**, a **Relational Memory-Driven Transformer Decoder**, and an **Explainable AI (Grad-CAM & Anatomical Finding Attribution)** suite.

---

## Table of Contents

- [Problem Statement & Research Objectives](#problem-statement--research-objectives)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Project Folder Structure](#project-folder-structure)
- [Installation & Setup](#installation--setup)
- [Dataset Preparation](#dataset-preparation)
  - [IU X-Ray (Indiana University)](#1-iu-x-ray-indiana-university-chest-x-ray)
  - [MIMIC-CXR (PhysioNet)](#2-mimic-cxr)
- [How to Run](#how-to-run)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Testing](#3-evaluation--testing)
  - [4. Single Image Inference & Demo](#4-single-image-inference--demo)
  - [5. Visualizing Attention Maps](#5-visualizing-attention-maps)
  - [6. Explainable AI (Grad-CAM & Anatomical Grounding)](#6-explainable-ai-grad-cam--anatomical-grounding)
  - [7. Clinical Efficacy (CE / MLC) Evaluation](#7-clinical-efficacy-ce--mlc-evaluation)
- [Experimental Results & Evaluation](#experimental-results--evaluation)
  - [NLG Benchmark Scores](#natural-language-generation-nlg-metrics)
  - [Semantic Similarity & BERTScore](#semantic-similarity--bertscore)
  - [Qualitative Example](#qualitative-sample-comparison)
- [References & Citation](#references--citation)
- [License & Acknowledgments](#license--acknowledgments)

---

## Problem Statement & Research Objectives

Chest radiography is the most ubiquitous medical imaging examination worldwide, utilized to diagnose pulmonary infections, cardiomegaly, pleural conditions, fractures, and masses. However:
1. **Radiologist Workload & Diagnostic Fatigue:** The sheer volume of chest radiographs creates severe cognitive overload for radiologists, resulting in report backlogs and diagnostic discrepancies.
2. **Repetition & Factual Hallucination in Standard NLG:** Conventional image captioning models (CNN-RNN or standard Vision Transformers) often suffer from severe dataset bias, generating repetitive boilerplate statements ("the lungs are clear", "heart is normal") while hallucinating non-existent findings or omitting subtle focal pathologies.
3. **Multi-View Information Disparity:** Radiologists routinely evaluate orthogonal projections (Posterior-Anterior/Anterior-Posterior and Lateral views). Typical models either only inspect a single view or fail to model geometric alignment across orthogonal views.

### Research Objectives
- **Multi-View Visual Extraction:** Extract and fuse representations across orthogonal projection pairs (Frontal + Lateral views) to mirror real-world radiologist workflows.
- **Deformable & Multi-Scale Dilated Modeling:** Capture complex, non-rigid lesion morphology using deformable convolutions alongside multi-scale dilated convolutions.
- **Relational Memory Transformer Decoding:** Retain and update key medical observations across token generation steps via relational memory slots, preventing repetition and contradictory statements.
- **Multi-Level Evaluation:** Evaluate outputs across lexical overlap (BLEU, ROUGE), deep contextual embeddings (BERTScore, BioBERT cosine similarity), and clinical efficacy (CE) multi-label classification.
- **Explainability & Spatial Grounding:** Implement Grad-CAM localization, anatomical finding attribution, and cross-attention mapping to verify that generated diagnoses correlate with actual visual pathologies.

---

## Key Features

- **Multi-View Dual Image Processing:** Tailored for datasets like IU X-Ray with dual-view support (Frontal + Lateral views) fused through spatial feature concatenation.
- **Hybrid Visual Feature Extractor:**
  - Standard ResNet-101 / DenseNet-121 backbone for hierarchical visual representations.
  - `DeformableNetwork`: Deformable 3x3 convolutions applied to intermediate layers (Conv3) to model irregular lesion contours.
  - `SemanticNetwork`: Dilated convolutions with multiple receptive field rates (dilation rates 6, 12, 18) on Conv5 features for multi-scale semantic context.
- **Memory-Driven Transformer Decoder:**
  - Relational Memory core (`rm_num_slots=3`, `rm_num_heads=8`, `rm_d_model=512`) interacting with multi-head attention states to preserve clinical narrative history.
  - Trigram blocking and beam search decoding (`beam_size=3`) to eliminate repetitive phrase loops.
- **Domain-Specific Medical Tokenization:**
  - Clean regex-based medical punctuation standardizer and thresholded vocabulary generation.
  - Seamless support for BioBERT and ClinicalBERT subword tokenization.
- **Explainable AI (XAI) Suite:**
  - **Grad-CAM Visual Heatmaps:** Visual class activation maps targeting specific radiological findings (pneumothorax, pleural effusion, cardiomegaly, consolidation, atelectasis, edema, nodules, fractures).
  - **Negation & Context Handling:** Rule-based negation parser distinguishing between positive findings and negative affirmations (e.g., *"no evidence of consolidation"* vs *"patchy consolidation present"*).
  - **Anatomical Grounding:** Extracts anatomical contexts (apical, basal, perihilar, peripheral, right/left lung lobes).
  - **Cross-Attention Heatmap Inspection:** Per-token cross-attention weight extraction over spatial feature grids (`main_plot.py`).
- **Clinical Efficacy (CE) Metric Pipeline:** Computes CheXpert-style disease-level classification metrics (AUC-ROC, Macro/Micro F1, Precision, Recall).

---

## Model Architecture

```
                    ┌────────────────────────────┐
                    │     Multi-View X-Rays      │
                    │  [Frontal + Lateral Views] │
                    └──────────────┬─────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌───────────────────────┐               ┌───────────────────────┐
   │ ResNet-101 Backbone   │               │ ResNet-101 Backbone   │
   │      (Frontal)        │               │       (Lateral)       │
   └──────────┬────────────┘               └───────────┬───────────┘
              │                                        │
              ▼                                        ▼
   ┌───────────────────────┐               ┌───────────────────────┐
   │ Deformable & Semantic │               │ Deformable & Semantic │
   │  Dilated Convolutions │               │  Dilated Convolutions │
   └──────────┬────────────┘               └───────────┬───────────┘
              │                                        │
              └────────────────────┬───────────────────┘
                                   │  Feature Concatenation
                                   ▼
                      ┌────────────────────────┐
                      │ Visual Patch & Average │
                      │      Vector Pools      │
                      └────────────┬───────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    Transformer Encoder        │
                   │ (Multi-Head Self-Attention)   │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │  Relational Memory Decoder    │
                   │  - Relational Memory Slots    │ ◄─── Interaction
                   │  - Masked Self-Attention      │      over decoded
                   │  - Multi-Head Cross-Attention │      steps
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ Generated Diagnostic Report   │
                   │    (Beam Search / Sampling)   │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │     Explainable AI (XAI)      │
                   │  - Grad-CAM Heatmaps          │
                   │  - Anatomical Finding Binding │
                   │  - Cross-Attention Weights    │
                   └───────────────────────────────┘
```

---

## Tech Stack & Dependencies

| Category | Libraries / Tools |
| :--- | :--- |
| **Deep Learning** | `PyTorch`, `Torchvision` |
| **Pretrained Models & NLP** | Hugging Face `transformers` (BioBERT / ClinicalBERT), `bert-score`, `nltk` |
| **Evaluation Metrics** | `scikit-learn` (AUC-ROC, F1, Precision, Recall), `pycocoevalcap` (BLEU 1-4, ROUGE-L) |
| **Computer Vision & Image** | `OpenCV (cv2)`, `Pillow (PIL)`, `matplotlib` |
| **Data & Scientific Computing**| `pandas`, `numpy`, `tqdm` |
| **Environment** | Python 3.8+, CUDA 11.7+ (Recommended for training) |

---

## Project Folder Structure

```
MedGen/
│
├── data/                                 # Dataset annotations and specialized XAI scripts
│   └── iu_xray/
│       ├── annotation.json               # Train/val/test splits with report text & image IDs
│       ├── images/                       # Directory containing IU X-ray image folders (untracked)
│       └── xai.py                        # Explainable AI: Grad-CAM, finding extractor & anatomical localization
│
├── models/                               # Core neural network architectures
│   ├── model.py                          # MedGenModel supporting single-view & dual-view datasets
│   └── r2gen.py                          # R2GenModel (dual-view forward implementation)
│
├── modules/                              # Modular components & neural network building blocks
│   ├── att_model.py                      # Base attention model abstraction
│   ├── caption_model.py                  # Captioning utilities & beam search algorithms
│   ├── dataloaders.py                    # R2DataLoader with data augmentations
│   ├── datasets.py                       # IuxrayMultiImageDataset for paired multi-view retrieval
│   ├── encoder_decoder.py                # Transformer encoder-decoder with Relational Memory (RM)
│   ├── loss.py                           # Cross-entropy language modeling loss
│   ├── metrics.py                        # NLP evaluation (BLEU, ROUGE-L) & clinical efficacy (MLC)
│   ├── optimizers.py                     # Adam optimizer & StepLR schedulers
│   ├── tester.py                         # Testing pipeline with report generation & export
│   ├── tester_plot.py                    # Cross-attention heatmap plotting routines
│   ├── tokenizers.py                     # Custom medical text tokenizer with vocabulary building
│   ├── trainer.py                        # Core training loop with validation & early stopping
│   ├── trainer_clinicalbert.py           # Alternative tokenizer integration for BioBERT/ClinicalBERT
│   ├── visual_extractor.py               # Standard visual extractor with average pooling
│   └── visual_extractor_new.py           # Hybrid visual extractor (Deformable + Dilated Convolutions)
│
├── pycocoevalcap/                        # MS COCO Caption evaluation suite (Python 3 compatible)
│   ├── bleu/                             # BLEU 1-4 computation
│   ├── rouge/                            # ROUGE-L computation
│   ├── meteor/                           # METEOR evaluation
│   ├── cider/                            # CIDEr evaluation
│   └── eval.py                           # Benchmark evaluation wrapper
│
├── results/                              # Checkpoint and evaluation output directory
│   └── iu_xray/                          # Checkpoints (model_best.pth, current_checkpoint.pth)
│
├── main_train.py                         # CLI script for model training
├── main_test.py                          # CLI script for model testing on test set
├── main_plot.py                          # Script to plot spatial attention heatmaps
├── generate_report.py                    # CLI script for single-image inference
├── infer_report.py                       # Standalone inference & side-by-side visualization
├── compute_ce.py                         # Evaluator for Clinical Efficacy (CE / MLC) multi-label metrics
├── requirements.txt                      # Project dependency specification
├── .gitignore                            # Repository ignore rules
└── README.md                             # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MedGen.git
cd MedGen
```

### 2. Create and Activate a Virtual Environment
```bash
# Using conda (recommended)
conda create -n medgen python=3.9 -y
conda activate medgen

# Or using venv
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Resources
Run the following in Python to download tokenizers and stopwords:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

---

## Dataset Preparation

### 1. IU X-Ray (Indiana University Chest X-Ray)
- **Source:** The Indiana University Chest X-Ray dataset (Open-i, NIH) contains 7,470 chest radiographs associated with 3,955 radiology reports.
- **Views:** Each study typically includes a frontal (PA/AP) and lateral projection.
- **Download:** Download images and reports from [NIH Open-i](https://openi.nlm.nih.gov/) or through Kaggle mirrors (e.g., [raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)).
- **Folder Setup:**
  ```
  data/
  └── iu_xray/
      ├── annotation.json
      └── images/
          ├── CXR1_IM-0001/
          │   ├── 0.png
          │   └── 1.png
          └── ...
  ```
- **Annotation Format (`annotation.json`):**
  ```json
  {
    "train": [
      {
        "id": "CXR1000_IM-0003",
        "image_path": ["CXR1000_IM-0003/0.png", "CXR1000_IM-0003/1.png"],
        "report": "the heart is normal in size . the lungs are clear . no pneumothorax or pleural effusion ."
      }
    ],
    "val": [...],
    "test": [...]
  }
  ```

### 2. MIMIC-CXR
- **Source:** Large, publicly available dataset of 377,110 radiographs with free-text reports from Beth Israel Deaconess Medical Center.
- **Access:** Requires credentialed access on [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) and completion of the CITI "Data or Specimens Only Research" training.
- Set `--dataset_name mimic_cxr` when training on MIMIC-CXR.

---

## How to Run

### 1. Data Preprocessing
To inspect data tokenization, cleaner logic, and vocabulary distribution, refer to [`modules/Datapreprocessing.ipynb`](modules/Datapreprocessing.ipynb).

### 2. Model Training
Train the MedGen model using [`main_train.py`](main_train.py):

```bash
python main_train.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 100 \
    --visual_extractor resnet101 \
    --d_model 512 \
    --num_layers 3 \
    --num_heads 8 \
    --rm_num_slots 3 \
    --rm_num_heads 8 \
    --rm_d_model 512 \
    --lr_ve 5e-5 \
    --lr_ed 1e-4 \
    --step_size 50 \
    --gamma 0.1 \
    --save_dir results/iu_xray \
    --monitor_metric BLEU_4 \
    --early_stop 50 \
    --n_gpu 1
```

**Key Arguments:**
- `--visual_extractor`: Backbone for image representation (e.g., `resnet101` or `densenet121`).
- `--rm_num_slots`: Number of relational memory slots (default: `3`).
- `--threshold`: Minimum word frequency cutoff to enter vocabulary.
- `--monitor_metric`: Metric used to track checkpoint updates (`BLEU_4` or `ROUGE_L`).
- `--early_stop`: Training patience (epochs without validation improvement before termination).

### 3. Evaluation & Testing
To evaluate a trained checkpoint on the test set and export predicted reports to CSV:

```bash
python main_test.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --load results/iu_xray/model_best.pth \
    --save_dir results/iu_xray \
    --beam_size 3
```
This produces `res.csv` (generated reports) and `gts.csv` (ground truth reports) in the `--save_dir`.

### 4. Single Image Inference & Demo
To generate a clinical report from a single chest X-ray image:

```bash
python generate_report.py \
    --image_path data/iu_xray/images/CXR1000_IM-0003/0.png \
    --model_path results/iu_xray/model_best.pth
```

Or run the interactive visual inference script:
```bash
python infer_report.py
```
This displays the input radiograph alongside the generated report using `matplotlib`.

### 5. Visualizing Attention Maps
To visualize cross-attention weights projected back onto spatial image patches per generated token:

```bash
python main_plot.py \
    --load results/iu_xray/model_best.pth \
    --save_dir results/iu_xray
```

### 6. Explainable AI (Grad-CAM & Anatomical Grounding)
Run the explainability engine in [`data/iu_xray/xai.py`](data/iu_xray/xai.py) to:
- Generate Grad-CAM heatmaps for key findings (`pneumothorax`, `pleural_effusion`, `cardiomegaly`, `consolidation`, `edema`, `atelectasis`).
- Detect negations (distinguish "no consolidation" from "focal consolidation").
- Map anatomical locations (apices, bases, perihilar, lower/upper lobes).

```bash
python data/iu_xray/xai.py
```

### 7. Clinical Efficacy (CE / MLC) Evaluation
Compute CheXpert-style multi-label disease classification metrics (Macro/Micro F1, Precision, Recall, AUC-ROC):

```bash
python compute_ce.py
```

---

## Experimental Results & Evaluation

The MedGen architecture was evaluated on the **IU X-Ray** benchmark test split (591 paired examination test studies).

### Natural Language Generation (NLG) Metrics

| Evaluation Split | Epoch | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Best** | 46 | **0.4262** | **0.2686** | **0.1892** | **0.1432** | **0.3285** |
| **Test Set Best** | 24 | **0.4498** | **0.2849** | **0.2047** | **0.1561** | **0.3535** |
| **Test Set (Final Converged)** | 97 | 0.4185 | 0.2506 | 0.1686 | 0.1212 | 0.3149 |

### Semantic Similarity & BERTScore

Beyond lexical n-gram matching, semantic congruence between generated reports and radiologist ground truth was validated using deep contextual embedding models:

| Evaluation Metric | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **BERTScore** | **0.8612** | **0.8796** | **0.8700** | Evaluated on full test set |
| **BioBERT Cosine Similarity** | — | — | **0.9517 ± 0.020** | Mean similarity across 590 test samples (Min: 0.8597, Max: 1.000) |

### Qualitative Sample Comparison

| Metric | Radiologist Ground Truth Report | MedGen Generated Report |
| :--- | :--- | :--- |
| **Sample 1** | *normal cardiomediastinal silhouette . there is no focal consolidation . there are no xxxx of a large pleural effusion . there is no pneumothorax . there is no acute bony abnormality seen .* | *the heart is normal in size . the pulmonary vascularity is within normal limits in appearance . no focal air space opacities . no pleural effusions or pneumothorax . no acute bony abnormalities .* |
| **Sample 2** | *lungs are clear . there is no pneumothorax or pleural effusion . the heart and mediastinum are within normal limits . bony structures are intact .* | *the cardiomediastinal silhouette is within normal limits for size and contour . the lungs are normally inflated without evidence of focal airspace disease pleural effusion or pneumothorax .* |
| **Sample 3** | *cardiac and mediastinal contours are within normal limits . the lungs are clear . bony structures are intact .* | *the lungs are clear . there is no pleural effusion or pneumothorax . the heart and mediastinum are normal . there is mild scoliosis of the spine .* |

---

## References & Citation

If you use MedGen or the ideas presented here in your research, thesis, or clinical benchmarks, please cite the following foundational works:

```bibtex
@inproceedings{chen2020generating,
  title={Generating Radiology Reports with Memory-driven Transformer},
  author={Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={2988--2998},
  year={2020}
}

@article{demner2016preparing,
  title={Preparing a collection of radiology examinations for distribution and retrieval},
  author={Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and Thoma, George R and McDonald, Clement J},
  journal={Journal of the American Medical Informatics Association},
  volume={23},
  number={2},
  pages={304--310},
  year={2016}
}

@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and Pollard, Tom J and Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and Peng, Yifan and Lu, Zhiyong and Mark, Roger G and Berkowitz, Seth J and Horng, Steven},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={317},
  year={2019}
}

@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedaldi, Andrea and Parikh, Devi and Batra, Dhruv},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={618--626},
  year={2017}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

---

## License & Acknowledgments

- **Caption Evaluation Code:** The `pycocoevalcap` subpackage is modified from [tylin/coco-caption](https://github.com/tylin/coco-caption) (BSD / Apache 2.0).
- **Medical Datasets:** Use of IU X-Ray and MIMIC-CXR is subject to their respective terms of service and research use agreements.

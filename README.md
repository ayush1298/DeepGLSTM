# DeepGLSTM: Deep Graph Convolutional Network and LSTM-based Approach for Predicting Drug-Target Binding Affinity

DeepGLSTM is an end-to-end deep learning framework designed to predict the binding affinity between drugs (small molecules) and target proteins. Originally developed to aid drug repurposing for SARS-CoV-2, the project has since evolved to include state-of-the-art transformer-based extensions.

## Project Overview

### Phase 1: The Original DeepGLSTM Work
The foundational work introduces a novel deep learning architecture combining **Graph Convolutional Networks (GCN)** for extracting drug molecular graph features and **Long Short-Term Memory (LSTM)** networks for extracting protein sequence features. 

Tested on Davis, KIBA, DTC, Metz, ToxCast, and STITCH datasets, the model effectively predicts binding affinity and was successfully used to rank 2,304 FDA-approved drugs against SARS-CoV-2 viral proteins for drug repurposing. For more details on the foundational methodology, please read the [original publication](https://arxiv.org/pdf/2201.06872v1.pdf).

### Phase 2: Modern Extensions (ESM-2, Attention Fusions, and GraphSite)
To push the predictive capabilities further, the codebase has been significantly extended:
- **ESM-2 Protein Embeddings:** We replaced the classic LSTM protein encoder with Facebook's massive pre-trained `ESM-2` language model (`esm2_t33_650M_UR50D`), allowing the extraction of much richer protein sequence representations.
- **Advanced Fusion Mechanisms:** Moving beyond simple vector concatenation, we implemented **Multi-Head Attention modules** (`self`, `cross`, and `both` configurations) to dynamically fuse drug and target embeddings before the final prediction stages.
- **Interactive Visualizations:** We built a web-based **Gradio Dashboard** (`demo.py`) that visually unpacks the model pipeline. It generates dynamic heatmaps of the Query/Key/Value steps within the cross-attention layer to explain *how* the model scores specific drug-target pairs.

## Model Architectures

### Original DeepGLSTM Structure
Below is the architectural flow of the original model relying on Graph Convolutions and LSTMs.
![DeepGLSTM Architecture](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/architecture.jpg "DeepGLSTM")

### ESM-2 + GCN Attention Flow
In our extended architecture, the flow is adapted to leverage modern transformers:
- **Drug Branch:** Molecular Graph -> GCN Layers -> Global Mean Pool -> Attention Projection
- **Protein Branch:** FASTA Sequence -> ESM-2 Model -> EOS Token Extraction -> Attention Projection
- **Fusion Layer:** Multi-Head Attention (`cross` and `self`) dynamically weighs the combined features before passing them to the final dense MLP layers for affinity scoring.

---

## Getting Started and Usage

The codebase is engineered to be highly flexible, allowing you to easily switch between the classic LSTM methodology and the new ESM-2 transformer models.

For complete instructions on:
- **Environment setup** (PyTorch, PyG, RDKit, Transformers)
- **Dataset generation** (Processing Davis, KIBA, etc., into `.pt` files)
- **Training pipelines & Ablation Studies** (Running subsets, automated scripts)
- **Running the Interactive Attention Dashboard**

**[Please Refer to instructions.md](instructions.md)**

---

## Datasets and Pretrained Models

The framework has been evaluated on multiple drug-target interaction datasets. Below are the download links for the raw datasets and original pretrained model checkpoints.

### Dataset Downloads
| Dataset | Dataset Download Links |
| :--- | :--- |
| **Davis** | [Google Drive Link](https://drive.google.com/drive/folders/1IDDOEAeBz3DiVWuwPDbGBm3-zJoY5S5L?usp=share_link) |
| **KIBA** | [Google Drive Link](https://drive.google.com/drive/folders/1LPPhV2RNhADE0rC5OKkHLluGD-T4yFUS?usp=share_link) |
| **DTC** | [Google Drive Link](https://drive.google.com/drive/folders/12iB06YOTsF7NTMhOcaF0f11jTjgmGJ9O?usp=share_link) |
| **Metz** | [Google Drive Link](https://drive.google.com/drive/folders/1_JNDEfFO8DFfyvVX633mv2mj43CG7Pnj?usp=share_link) |
| **ToxCast** | [Google Drive Link](https://drive.google.com/drive/folders/1PcFlVYdq4EJuHAF8vG7x2FntrPNHt69m?usp=share_link) |
| **STITCH** | [Google Drive Link](https://drive.google.com/drive/folders/1F4sRWS9k4bbs3sDf_bPpxiCnpYcTeSXf?usp=share_link) |

> *Note:* Store downloaded datasets in the `data/` folder. Each link contains a `_train.csv` and `_test.csv` file.

---

## Model Performance Stats

Plots showing original DeepGLSTM predictions versus measured binding affinity values. **Coef_V** refers to the Pearson correlation coefficient.

![Model Performance](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/Full_fig%20.jpg "Full_fig")

*(a) Davis (b) KIBA (c) DTC (d) Metz (e) ToxCast (f) STITCH*

## Case Studies on SARS-CoV-2 Viral Proteins
We applied DeepGLSTM to calculate a Combined Score for 2,304 FDA-approved drugs against 5 viral proteins to identify ideal repurposing candidates.

![Case Study Table 1](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/Sup_table.jpeg "Sup_1")
![Case Study Table 2](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/sup_table2.jpeg "Sup_2")

# NERO: Explainable Out-of-Distribution Detection with Neuron-level Relevance
This codebase provides a Pytorch implementation of:

📄Paper Available: https://arxiv.org/abs/2506.15404

🎉 Accepted to MICCAI 2025
<!-- 
[![NERO](https://img.shields.io/badge/)](https://arxiv.org/abs/2506.15404)
*Chhetri A.*, Korhonen J, Gyawali P,  Bhattarai B. -->
## Abstract
Ensuring reliability is paramount in deep learning, particularly within the domain of medical imaging, where diagnostic decisions often hinge on model outputs. The capacity to separate out-of-distribution (OOD) samples has proven to be a valuable indicator of a model’s reliability in research. In medical imaging, this is especially critical, as identifying OOD inputs can help flag potential anomalies that might otherwise go undetected. While many OOD detection methods rely on feature or logit space representations, recent works suggest these approaches may not fully capture OOD diversity. To address this, we propose a novel OOD scoring mechanism, called NERO, that leverages neuron-level relevance at the feature layer. Specifically, we cluster neuron-level relevance for each in-distribution (ID) class to form representative centroids and introduce a relevance distance metric to quantify a new sample’s deviation from these centroids, enhancing OOD separability. Additionally, we refine performance by incorporating scaled relevance in the bias term and combining feature norms. Our framework also enables explainable OOD detection. We validate its effectiveness across multiple deep learning architectures on the gastrointestinal imaging benchmarks Kvasir and GastroVision, achieving improvements over state-of-the-art OOD detection methods.

🧠 Key Contributions
---
1. We propose a novel explainable post-hoc OOD detection method, NERO,
that leverages neuron-level relevance to analyze prediction-relevant patterns.
2. We demonstrate the effectiveness of our method through extensive empiri-
cal evaluations on challenging medical image benchmarks, showing superior
performance compared to state-of-the-art OOD detection techniques.

<!-- ## 🧪 Example Scripts for Training and Inference
To get started, clone the repository:
```
git clone https://github.com/anju-chhetri/NERO.git
cd NERO
``` -->
## 📂 Dataset Sources
This project uses two publicly available medical imaging datasets for evaluating OOD detection methods: [KvasirV2](https://datasets.simula.no/kvasir/) and [Gastrovision](https://github.com/DebeshJha/GastroVision)


## 🧪 Example Scripts for Training and Inference
To get started, clone the repository:
```
git clone https://github.com/bhattarailab/NERO.git
cd NERO
```
## 📂 Dataset Sources
This project uses two publicly available medical imaging datasets for evaluating OOD detection methods: [KvasirV2](https://datasets.simula.no/kvasir/) and [Gastrovision](https://github.com/DebeshJha/GastroVision)

### 💾 Pre-trained Checkpoints

Pre-trained weights of the classification networks are included in the repository: ResNet-18 and DeiT models trained on KvasirV2 and GastroVision datasets can be found [here for Kvasir - ResNet-18](https://github.com/bhattarailab/NERO/blob/main/checkpoints/resnet18/kvasir.pt), [GastroVision - ResNet-18](https://github.com/bhattarailab/NERO/blob/main/checkpoints/resnet18/gastrovision.pt), [Kvasir - DeiT](https://github.com/bhattarailab/NERO/blob/main/checkpoints/deit/kvasir.pt), and [GastroVision - DeiT](https://github.com/bhattarailab/NERO/blob/main/checkpoints/deit/gastrovision.pt).

## 🚀 Running OOD Detection

To run OOD detection on the **KvasirV2** dataset:

```bash
bash ood/demo_kvasir.sh
```
To run it on the **GastroVision** dataset:
```bash
bash ood/demo_gastrovision.sh
```

``` bash
python3 ood/eval_ood.py \
    --in-dataset 'kvasir' \
    --num_classes 3 \
    --model-arch 'resnet18' \
    --weights 'path/to/model/checkpoints' \
    --seed 42 \
    --base-dir 'path/to/save/results' \
    --id_path_train 'path/to/in-distribution/training/data' \
    --id_path_valid 'path/to/in-distribution/validation/data' \
    --ood_path 'path/to/ood/data'

```
Modify the paths and options as needed for your dataset, model architecture, or checkpoint location.

## Results
# ResNet-18:

| Method        | Kvasir-v2 |      | GastroVision |      |
|---------------|-----------|------|--------------|------|
|               | AUC       | FPR95| AUC          | FPR95|
| MSP           | 90.3      | 41.72| 66.93        | 90.56|
| ODIN          | **91.77** | <u>35.44</u>| 69.79        | 79.27|
| Energy        | 88.85     | 52.36| 70.31        | 79.79|
| Entropy       | 90.38     | 41.86| 67.37        | 87.32|
| MaxLogit      | 88.9      | 52.38| 70.08        | 80.44|
| Mahalanobis   | 84.05     | 54.06| 65.93        | 89.69|
| ViM           | 90.62     | 41.1 | 72.70        | 76.98|
| NECO          | 89.64     | 47.90| **79.81**    | **71.61**|
| Energy+ReAct  | 86.57     | 53.78| 61.93        | 83.86|
| GradNorm      | 85.33     | 54.68| 62.55        | 90.5 |
| **NERO (ours)**| <u>90.76</u>    | **28.84**| <u>75.95</u>    | <u>74.33</u>|

# Deit
| Method        |  kvasir-V2   |      | GastroVision |      |
|---------------|------------|------|--------------|------|
|               | AUC        | FPR95| AUC          | FPR95|
| MSP           | 87.05      | 40.18| 70.0         | 90.74|
| ODIN          | 88.41      | 36.4 | 73.37        | 83.68|
| Energy        | 85.77      | 44.02| 75.35        | 83.68|
| Entropy       | 87.2       | 39.94| 70.34        | 90.19|
| MaxLogit      | 85.77      | 44.02| 75.11        | 84.2 |
| Mahalanobis   | **94.50**  | <u>21.86</u>| 75.68        | 81.43|
| ViM           | <u>93.88</u>      | 24.38| 76.69        | <u>78.37</u>|
| NECO          | 88.31      | 37.60| <u>76.95</u>        | 81.92|
| Energy+ReAct  | 83.49      | 46.84| 73.42        | 83.22|
| GradNorm      | 71.33      | 57.8 | 54.85        | 88.68|
| NERO (ours)   | 92.73      | **18.96**| **82.03**| **76.74**|

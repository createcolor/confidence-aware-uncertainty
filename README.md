# Improving Uncertainty Estimation with Confidence-Aware Training Data
\[[pdf](https://openaccess.thecvf.com/content/WACV2025/papers/Korchagin_Improving_Uncertainty_Estimation_with_Confidence-Aware_Training_Data_WACV_2025_paper.pdf)\] \[[supp](https://openaccess.thecvf.com/content/WACV2025/supplemental/Korchagin_Improving_Uncertainty_Estimation_WACV_2025_supplemental.pdf)\] \[[data](https://zenodo.org/records/14285035)\]

The official implementation of the paper "Improving Uncertainty Estimation with Confidence-Aware Training Data" by Korchagin S., et al. The work was presented at the 2025 Winter Conference on Applications of Computer Vision.

## Abstract
AI-driven second-opinion systems play a crucial role in decision-making especially in medicine where accurate predictions guide clinicians. However quantifying uncertainty in deep learning is challenging as current methods often rely on hard class labels which do not reflect true prediction confidence. This often results in overconfident predictions and slow convergence to true probabilities. To address this we suggest a new method that separates uncertainty into two types: epistemic and aleatoric. We estimate these uncertainties using hard and soft confidence labels with experts providing confidence levels that indicate the likelihood of misclassification. We release an updated blood typing dataset consisting of 3139 images with soft labels of uncertainty annotations from six experts and hard labels collected from medical records. Proposed approach improves SotA uncertainty estimation quality by two times for blood typing (classification) and by 62% for histology (segmentation).

## Installation and Usage
The code was run on `Python 3.10`. To install all necessary dependencies, run
```
pip install -r requirements.txt
```

The code is split in three parts:
* All code and instructions to validate experiments for the blood typing task (classification) are located in the `uncertaint_classification` directory.
* All code and instructions to validate experiments for the lung CT scan segmentation and retinal fundus image segmentation tasks are located in the `segmentation` directory.
* All code and instructions to validate experiments on synthetic data will be made available soon.

## Data
The blood typing BloodyWell dataset is available [here](https://zenodo.org/records/14285035). The `markup/BloodyWell` directory stores additional metadata that was used for training and testing the models.


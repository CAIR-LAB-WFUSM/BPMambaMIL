# BPMambaMIL: A Bio-inspired Prototype-guided Multiple Instance Learning for Oncotype DX Risk Assessment in Histopathology

[![CMPB](https://img.shields.io/badge/Journal-Computer_Methods_and_Programs_in_Biomedicine-blue)](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine)

> **🎉 News:** Our paper has been accepted by **Computer Methods and Programs in Biomedicine**.

## 📝 Abstract

Breast cancer remains one of the most prevalent malignancies among women, with hormone receptor-positive (HR+)/human epidermal growth factor receptor 2-negative (HER2–) breast cancers constituting a majority, with treatment decisions often guided by genomic assays such as the 21-gene recurrence score assay, Oncotype DX. Although Oncotype DX provides critical prognostic and predictive insights, its high cost and limited accessibility create substantial barriers, especially for patients with constrained financial resources. 

To reduce the test cost, we aim to leverage H&E-stained whole slide images (WSIs) to predict Oncotype DX risk. Since WSIs are extremely large and contain redundant information, directly processing them is both computationally expensive and prone to errors. To address these limitations, we introduce a bio-inspired prototype-guided model (**BPMambaMIL**), a novel weakly supervised learning framework that integrates the Mamba mechanism with prototypical guidance to predict Oncotype DX score intervals directly from pathology images. 

Our model was evaluated on an in-house dataset with clinical Oncotype DX scores, where it achieved an **AUC of 0.839**, a **5.61% improvement** over the baseline model (MambaMIL), and demonstrated robust predictive performance, particularly in identifying high-risk score ranges (**accuracy: 0.714 vs 0.419**). Further assessments on two public breast cancer pathology image datasets using six state-of-the-art models underscored BPMambaMIL’s generalizability on research-based ODX scores and binary tumor classification tasks. By evaluating various clinical scenarios, the proposed method not only enhances the accuracy of breast cancer recurrence risk predictions but also offers a cost-effective alternative to genomic assays, thus improving clinical outcomes.

## ⚙️ Installation

The environment setup relies on the MambaMIL framework. Please refer to the official repository below for installation instructions:

* **MambaMIL:** [isyangshu/MambaMIL](https://github.com/isyangshu/MambaMIL)
    * *[MICCAI 2024] Official Code for "MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology".*

*We thank the authors of MambaMIL for their great work.*

## 🚀 How to Train

### 1. Prepare Slide Features
Please refer to a general foundation model pipeline to extract features and place them in your local directory. 
* **Example extractor:** TRIDENT (or other foundation models).

### 2. Prepare Prototypes
Run the following command to generate prototypes and save them to your local directory. Adjust the arguments as necessary for your dataset.

```bash
python prototype.py \
  --datadf <path_to_data_dataframe> \
  --featdir <path_to_feature_directory> \
  --task osu35 \
  -t 4 \
  --psize 896 \
  --savedir ./output/osu35_ctp_eefo_47 \
  --algorithm eefo \
  --seed 47 \
  --gpu \
  --batch-size 100
```

### 3. Training and Testing
Please use the provided shell script for the full training, testing, and tuning pipeline. You can set customized arguments within the script to tailor the training to your specific model configuration.

```bash
sh run.sh
```

## 🤝 Acknowledgements

Huge thanks to the authors of the following open-source projects:
* [TRIDENT](https://github.com/Startcodeforu/Trident)
* [MambaMIL](https://github.com/isyangshu/MambaMIL)

## 📄 License & Citation

If you find our work useful in your research, please consider citing our paper:

**Yongxin Guo, Ziyu Su, Onur C. Koyun, Hao Lu, Robert Wesolowski, Gary Tozbikian, M. Khalid Khan Niazi, Metin N. Gurcan**, "BPMambaMIL: A bio-inspired prototype-guided multiple instance learning for oncotype DX risk assessment in histopathology", *Computer Methods and Programs in Biomedicine*, Volume 272, 2025, 109039, ISSN 0169-2607.
[https://doi.org/10.1016/j.cmpb.2025.109039](https://doi.org/10.1016/j.cmpb.2025.109039)

### BibTeX
```bibtex
@article{guo2025bpmambamil,
  title = {BPMambaMIL: A bio-inspired prototype-guided multiple instance learning for oncotype DX risk assessment in histopathology},
  author = {Yongxin Guo and Ziyu Su and Onur C. Koyun and Hao Lu and Robert Wesolowski and Gary Tozbikian and M. Khalid Khan Niazi and Metin N. Gurcan},
  journal = {Computer Methods and Programs in Biomedicine},
  volume = {272},
  pages = {109039},
  year = {2025},
  issn = {0169-2607},
  doi = {10.1016/j.cmpb.2025.109039},
  url = {[https://www.sciencedirect.com/science/article/pii/S0169260725004560](https://www.sciencedirect.com/science/article/pii/S0169260725004560)}
}
```

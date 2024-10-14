![image](https://github.com/user-attachments/assets/99f73d68-6ac4-4431-abe4-dec3b565a2fc)# CAT: Coordinating Anatomical-Textual Prompts for Multi-Organ and Tumor Segmentation
 Zhongzhen Huang, [Yankai Jiang](https://scholar.google.com/citations?user=oQKcL_oAAAAJ), [Rongzhao Zhang](https://scholar.google.com/citations?user=NMp31uMAAAAJ), [Shaoting Zhang](https://scholar.google.com/citations?user=oiBMWK4AAAAJ), [Xiaofan Zhang](https://scholar.google.com/citations?user=30e95fEAAAAJ)

  <p align="center">
    <a href='https://arxiv.org/abs/2406.07085'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/zongzi3zz/CAT/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://www.youtube.com/watch?v=WI-65Jk0j50'>
      <img src='https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=YouTube' alt='Video'>
    </a>
  </p>
<br />

## 🛠️ Quick Start

### Installation

- It is recommended to build a Python-3.10 virtual environment using conda

  ```bash
  conda create --name cat-env python=3.10 -y
  conda activate mgllava-env
  git clone https://github.com/zongzi3zz/CAT.git
  cd CAT
  pip install -r requirement.txt

### Dataset Preparation
- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details)
- 05 [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block)
- 06 [Liver segmentation (3D-IRCADb)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
- 07 [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD)
- 08 [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- 09 [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org)
- 10 [Decathlon (Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 11 [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
- 12 [AbdomenCT 12organ](https://zenodo.org/records/7860267)

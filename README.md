# CAT: Coordinating Anatomical-Textual Prompts for Multi-Organ and Tumor Segmentation
 [Zhongzhen Huang](https://scholar.google.com/citations?user=LrZdFHgAAAAJ), [Yankai Jiang](https://scholar.google.com/citations?user=oQKcL_oAAAAJ), [Rongzhao Zhang](https://scholar.google.com/citations?user=NMp31uMAAAAJ), [Shaoting Zhang](https://scholar.google.com/citations?user=oiBMWK4AAAAJ), [Xiaofan Zhang](https://scholar.google.com/citations?user=30e95fEAAAAJ)

  <p align="center">
    <a href='https://arxiv.org/abs/2406.07085'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/zongzi3zz/CAT/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://youtu.be/pLiBnWpk5iY'>
      <img src='https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=YouTube' alt='Video'>
    </a>
  </p>
<br />

## 🎉 News
- **\[2024/09\]** CAT is accepted to NeurIPS 2024!

## 🛠️ Quick Start

### Installation

- It is recommended to build a Python-3.9 virtual environment using conda

  ```bash
  git clone https://github.com/zongzi3zz/CAT.git
  cd CAT
  conda env create -f env.yml

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
### Dataset Pre-Process
1. Please refer to [CLIP-Driven](https://github.com/ljwztc/CLIP-Driven-Universal-Model) to organize the downloaded datasets.
2. Modify [ORGAN_DATASET_DIR](https://github.com/zongzi3zz/CAT/blob/2146b2e972d0570956c52317a75c823891a4df2c/label_transfer.py#L51) and [NUM_WORKER](https://github.com/zongzi3zz/CAT/blob/2146b2e972d0570956c52317a75c823891a4df2c/label_transfer.py#L53) in label_transfer.py  
3. `python -W ignore label_transfer.py`
4. The example of data configure for training and evaluation can be seen in [datalist](https://github.com/zongzi3zz/CAT/tree/main/datalist)
### Prompt Feats
We provide the prompt feats in [BaiduNetdisk](https://pan.baidu.com/s/1JoR-esWoRrFrxsSpacq7bw?pwd=mbae) (code: `mbae`).
### Model Weights
The weights used for train and inference are provided in [GoogleDrive](https://drive.google.com/drive/folders/1fpFyF4F4HrpwKNpPDeH7nXrjo89LC0CG?usp=drive_link).
| Data | Download |
|  ----   |  ----  |
| Partial  | [link](https://drive.google.com/file/d/1wx9lIRfMnktkP0AcCbaBEDaiQgqafk9l/view?usp=drive_link) |
| Full | [link](https://drive.google.com/file/d/12aSwZduf-aS1GQ9Sn3St0Zwn_sIHwxhE/view?usp=drive_link) |

### Train & Evaluation
The entire training process takes approximately 4 days using 8×A100 GPUs.
- **Train Pipeline**: 
  Set the parameter [`data_root`](https://github.com/zongzi3zz/CAT/blob/7fe7477b773ca1d83e6f47ff8a7aa90b6a392dec/train_CAT.py#L236) and run:
  ```shell
  bash scripts/train.sh
  ```
We provide two model weights, hoping that the weights trained with full data would support organ and tumor segmentation tasks in other scenarios. 
Set the parameter `pretrain_weights` and run:
- **Evaluation**
  ```shell
  bash scripts/test.sh
  ```
- **Inference**
  ```shell
  bash scripts/inference.sh
  ```
If you want to use [the Full weight](https://drive.google.com/file/d/12aSwZduf-aS1GQ9Sn3St0Zwn_sIHwxhE/view?usp=drive_link), you need to add `--only_last`

## Citation
If you find CAT useful, please cite using this BibTeX:
```bibtex
@article{huang2024cat,
  title={CAT: Coordinating Anatomical-Textual Prompts for Multi-Organ and Tumor Segmentation},
  author={Huang, Zhongzhen and Jiang, Yankai and Zhang, Rongzhao and Zhang, Shaoting and Zhang, Xiaofan},
  journal={arXiv preprint arXiv:2406.07085},
  year={2024}
}
```
## Acknowledgement
- [CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model): the codebase we built upon.

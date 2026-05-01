# SplatHLoc
Official code for CVPR 2026 paper "Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting"

## 🛠️ Environment Setup
```
conda create -n splathloc python=3.8 -y
conda activate splathloc

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
pip install git+https://github.com/cvg/LightGlue.git
pip install submodules/simple-knn
```

## 📦 Data Preparation

#### 7-Scenes Dataset

1. Download images following HLoc.

```bash
export dataset=datasets/7scenes
for scene in chess fire heads office pumpkin redkitchen stairs; \
do wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip -P $dataset \
&& unzip $dataset/$scene.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done
```

2. Download full reconstructions
   from [visloc_pseudo_gt_limitations](https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/main?tab=readme-ov-file#full-reconstructions):

```bash
pip install gdown
gdown 1ATijcGCgK84NKB4Mho4_T-P7x8LSL80m -O $dataset/7scenes_reference_models.zip
unzip $dataset/7scenes_reference_models.zip -d $dataset
# move sfm_gt to each dataset
for scene in chess fire heads office pumpkin redkitchen stairs; \
do mkdir -p $dataset/$scene/sparse && cp -r $dataset/7scenes_reference_models/$scene/sfm_gt $dataset/$scene/sparse/0 ; done
```

#### 12-Scenes Dataset

1. Download images following HLoc.

```bash
export dataset=datasets/12scenes
for scene in apt1 apt2 office1 office2; \
do wget https://graphics.stanford.edu/projects/reloc/data/$scene.zip -P $dataset \
&& unzip $dataset/$scene.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done
```

2. Download full reconstructions
   from [visloc_pseudo_gt_limitations](https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/main?tab=readme-ov-file#full-reconstructions):

```bash
pip install gdown
gdown 1u5du-cYp3J3-BfybZVkhvgv0PPua8tud -O $dataset/12scenes_reference_models.zip
unzip $dataset/12scenes_reference_models.zip -d $dataset
# move sfm_gt to each dataset
for scene in apt1/kitchen apt1/living apt2/bed apt2/kitchen apt2/living apt2/luke office1/gates362 office1/gates381 office1/lounge office1/manolis office2/5a office2/5b; 
do mkdir -p "$dataset/$scene/data/sparse" && cp -r $dataset/12scenes_reference_models/$scene/sfm_gt $dataset/$scene/sparse/0/ ;
done
```

#### Cambridge Landmarks Dataset

1. Download images from PoseNet's project page:

```bash
export dataset=datasets/cambridge
export scenes=( "KingsCollege" "OldHospital" "StMarysChurch" "ShopFacade" "GreatCourt" )
export IDs=( "251342" "251340" "251294" "251336" "251291" )
for i in "${!scenes[@]}"; do
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/${IDs[i]}/${scenes[i]}.zip -P $dataset \
&& unzip $dataset/${scenes[i]}.zip -d $dataset ; done
```

2. Install Mask2Former:

```bash
cd submodules/Mask2Former
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
# download model
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl
cd ../..
```

3. Preprocess data:

```bash
bash scripts/dataset_preprocess.sh
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🚀 Training

```bash
# For 7-Scenes:
bash scripts/train_7s.sh
# For 12-Scenes:
bash scripts/train_12s.sh
# For Cambridge:
bash scripts/train_cam.sh
```

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>


## 📈 Evaluation

- Download and extract the pretrained Feature Guassian Map for [7-Scenes](https://drive.google.com/file/d/1tLvRUNylbMZ1oUQooHvUnKwf4YnBnfdj/view?usp=sharing), [12-Scenes](https://drive.google.com/file/d/1mJXtgPAVSXjFNVznI6O8UzsdavFLmFRi/view?usp=sharing) and [Cambridge](https://drive.google.com/file/d/1AEAFR0WeD6vBanQsxvgRQFrVO6g27GBC/view?usp=sharing) datasets into the `map/` folder.

- Download the pretrained MixVPR model from [official link](https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view), and place it under `vpr_model/` folder.

Reproduce the experimental results.
```bash
# For 7-Scenes:
bash scripts/test_7s.sh
# For 12-Scenes:
bash scripts/test_12s.sh
# For Cambridge:
bash scripts/test_cam.sh
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 📜 Citing

If you find SplatHLoc is useful in your research, please consider giving us a star 🌟 and citing it by the following BibTeX entry:

```
@article{tao2026hierarchical,
  title={Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting},
  author={Tao, Huaqi and Liu, Bingxi and Chen, Guangcheng and Tang, Fulin and He, Li and Zhang, Hong},
  journal={arXiv preprint arXiv:2603.29185},
  year={2026}
}
```
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🙏 Acknowledgement
Our work is primarily based on the following codebases: [HLoc](https://github.com/cvg/Hierarchical-Localization), [STDLoc](https://github.com/zju3dv/STDLoc), [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs), [MixVPR](https://github.com/amaralibey/MixVPR),  [JamMa](https://github.com/leoluxxx/JamMa). We are sincerely grateful for their works.

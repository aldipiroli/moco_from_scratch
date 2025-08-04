# MoCo from scratch
Implementing ["Momentum Contrast for Unsupervised Visual Representation Learning"](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) CVPR 2020 from scratch.

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/moco_from_scratch
pip install -r requirements.txt && cd moco
``` 
### Train 
``` 
python train.py config/moco_config.yaml
```
# Learning Gradient Fields for Shape Generation

This repository contains a PyTorch implementation of the paper:

[*Learning Gradient Fields for Shape Generation*](http://www.cs.cornell.edu/~ruojin/ShapeGF/)
[[Project page]](http://www.cs.cornell.edu/~ruojin/ShapeGF/)
[[Arxiv]](https://arxiv.org/abs/2008.06520)
[[Short-video]](https://www.youtube.com/watch?v=HQTbtFzDYAU)
[[Long-video]](https://www.youtube.com/watch?v=xCCdnzt7NPA)

[Ruojin Cai*](http://www.cs.cornell.edu/~ruojin/), 
[Guandao Yang*](https://www.guandaoyang.com/), 
[Hadar Averbuch-Elor](http://www.cs.cornell.edu/~hadarelor/), 
[Zekun Hao](http://www.cs.cornell.edu/~zekun/), 
[Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/), 
[Noah Snavely](http://www.cs.cornell.edu/~snavely/), 
[Bharath Hariharan](http://home.bharathh.info/)
_(* Equal contribution)_

ECCV 2020 (*Spotlight*)

<p float="left">
    <img src="assets/ShapeGF.gif" height="256"/>
</p>

## Dependencies
```bash
# Create conda environment with torch 1.11.0 and CUDA 11.3
conda env create -f environment.yml
conda activate ShapeGF

# Compile the evaluation metrics
cd evaluation/pytorch_structural_losses/
make clean
make all
```

## Dataset

Please unzip the dataset and save the dataset under /data directory: [link](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ). 

# Pretrained model

[link](https://drive.google.com/drive/folders/1kTBlrqeDSGbYQA45SOX7hMCzhU6GXEQu?usp=sharing)  
The epoch_199 one is recon, the other one is gen.  

## Training and Testing
python train.py configs/recon/airplane/airplane_recon_add.yaml  
python train.py configs/gen/airplane_gen_add.yaml  

python test.py configs/recon/airplane/airplane_recon_add.yaml \  
    --pretrained <directory of pretrained model.pt>  
python test.py configs/gen/airplane_gen_add.yaml \  
    --pretrained <directory of pretrained model.pt>  


# Original Cite 
Please cite our work if you find it useful: 
```bibtex
@inproceedings{ShapeGF,
 title={Learning Gradient Fields for Shape Generation},
 author={Cai, Ruojin and Yang, Guandao and Averbuch-Elor, Hadar and Hao, Zekun and Belongie, Serge and Snavely, Noah and Hariharan, Bharath},
 booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
 year={2020}
}
```
#### Original Acknowledgment
This work was supported in part by grants from Magic Leap and Facebook AI, and the Zuckerman STEM leadership program.


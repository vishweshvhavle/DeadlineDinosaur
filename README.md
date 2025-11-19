# DeadlineDinosaur

[Parag Sarvoday Sahu](https://paragsarvoday.github.io/),
[Vishwesh Vhavle](https://vishweshvhavle.github.io/),
[Kshitij Aphale](https://github.com/lighterbird),
and
[Avinash Sharma](https://3dcomputervision.github.io/about/)

## 1. Installation 

### Make sure the current working directory is the project folder
```
cd DeadlineDinosaur
```

### Create conda environment using `environment.yml` file
> Note that this can take quite a while.
```
conda env create --file environment.yml
conda activate DeadlineDinosaur
```

### Install all submodules
```
pip install deadlinedino/submodules/simple-knn
pip install deadlinedino/submodules/fused_ssim
pip install deadlinedino/submodules/gaussian_raster
```

### Structure
```
DeadlineDinosaur/
├── deadlinedino
│   ├── submodules
│   │   ├── fused_ssim
│   │   ├── gaussian_raster
│   │   └── simple-knn
├── data
├── outputs
├── environment.yml
├── train.py
├── evaluate.py
├── README.md
├── LICENSE.md
├── LiteGS_LICENSE.md
└── DashGaussian_LICENSE.md          
```

## 2. Data Preparation
```
mkdir data
```
Place the eval_data_pinhole/ directory inside data/

The data folder should look like:

```
data/
└── eval_data_pinhole             # Processed dataset folders (13 total)   
    ├── 1747834320424            
    │   ├── images_gt_downsampled # Extracted frames (.jpg)
    │   │   ├── 000000.png
    │   │   ├── ...
    │   │   └── 000199.png
    ├── sparse                    # Sparse reconstruction data
    │   │   └── 0
    │   │       ├── cameras.txt
    │   │       ├── frames.txt
    │   │       ├── images.txt
    │   │       ├── points3D.ply
    │   │       ├── points3D.txt
    │   │       ├── project.ini
    │   │       └── rigs.txt
    │   └── train_test_split.json
    ├── 1748153841908
    ├── 1748165890960
    ├── 1748242779841
    ├── 1748243104741
    ├── 1749449291156
    ├── 1749606908096
    ├── 1749803955124
    ├── 1750578027423
    ├── 1750824904001
    ├── 1750825558261
    ├── 1750846199351
    ├── 1751090600427
    └── ReadMe_Round2.md
```

## 3. Training
Run with default structure
```
python train.py
```
Run with custom paths and GPU
```
python train.py \
    --dataset_dir data/eval_data_pinhole \
    --output_dir outputs \
    --gpu 0
```

## 4. Evaluation
Run with default structure
```
python evaluate.py
```
Run with custom paths and GPU
```
python evaluate.py \
    --dataset_dir data/eval_data_pinhole \
    --output_dir outputs \
    --output_run_dir outputs \       # Optional: The script evalutes the most recent outputs directory by default.
    --gpu 0
```
> Note: The evaluation takes a while. The scene-wise rendered and gt images will be saved in the outputs directory. The scene-wise as well as average PSNRs will get printed after the entire evaluation has completed. The scene-wise training times are also saved as .json files along with the .ply files.

## Citation

If you find our code or paper useful, please consider citing
```bibtex
@misc{sahu2025deadlinedinosaur,
title={DeadlineDinosaur: Fast Gaussian Splatting for SIGGRAPH Asia's 3D Gaussian Splatting Challenge},
author={Parag Sarvoday Sahu, Vishwesh Vhavle, Kshitij Aphale, and Avinash Sharma},
year={2025},
url={https://github.com/paragsarvoday/SA_GS_Challenge},
}
```

## Contact

Contact [Parag Sarvoday Sahu](mailto:parag.sahu@iitgn.ac.in) for questions, comments and reporting bugs, or open a GitHub Issue.


## License 
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## 🙏 Acknowledgements

We acknowledge this work is based on DashGaussian (CVPR '25) and LiteGS

- [DashGaussian](https://github.com/YouyuChen0207/DashGaussian)
- [LiteGS](https://github.com/MooreThreads/LiteGS)


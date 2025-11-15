# DeadlineDinosaur

[Parag Sarvoday Sahu](https://paragsarvoday.github.io/),
[Vishwesh Vhavle](https://vishweshvhavle.github.io/),
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
conda activate DDinoGS
```

### Install all submodules
```
pip install deadlinedino/submodules/simple-knn
pip install deadlinedino/submodules/fused_ssim
pip install deadlinedino/submodules/gaussian_raster
pip install deadlinedino/submodules/lanczos-resampling
```

Structure (To be Updated*)
```
DeadlineDinosaur/
├── environmentyml    # Base dependencies only
├── setup_deadlinedinosaur.sh     # Complete setup script
├── compiled_packages/            # Precompiled extensions
│   ├── install_compiled.sh
│   ├── diff_gaussian_rasterization*
│   ├── simple_knn*
│   ├── FastLanczos*
│   ├── fused_ssim*
│   └── fastlanczos*
├── gaussian_splatting_compiled.tar.gz  # For easy distribution
├── environment.yml               # Original (build from source)
└── README.md                     
```

## 2. Preprocessing
```
mkdir data
```
Place the Dataset/ directory inside data

```
chmod +x process_dataset.sh 
./process_dataset.sh 
```

The data folder should look like:

```
data/
├── Dataset/
│   └── [timestamp]/              # Raw dataset folders (31 total)
│       ├── [timestamp]_flip.mp4  # Original video
│       └── inputs/               # Raw input data
│           ├── gravity.txt
│           ├── slam/
│           ├── traj_full.txt.bak
│           └── videoInfo.txt
└── ProcessedDataset/
    └── [timestamp]/              # Processed dataset folders (31 total)
        ├── images/               # Extracted frames (.jpg)
        ├── metadata/             # Processing metadata
        │   ├── gravity.txt
        │   └── trajectory.txt
        └── sparse/               # Sparse reconstruction data
            └── 0/
```

## 3. Training (To be Updated*)
Run with default structure
```
python run_all_scenes.py
```
Run with custom paths and GPU
```
python run_all_scenes.py \
    --dataset_dir data/ProcessedDataset \
    --output_dir outputs \
    --gpu 0
```

## 4. Evaluation (To be Updated*)
```
python calculate_average_metrics.py --output_dir outputs
```

Please find results in the generated results.json file in outputs/ directory

## Citation (To be Updated*)

If you find our code or paper useful, please consider citing
```bibtex
@misc{sahu2025deadlinedinosaur,
title={DeadlineDinosaur: Fast Gaussian Splatting for SIGGRAPH Asia's 3D Gaussian Splatting Challenge},
author={Parag Sahu, Vishwesh Vhavle, and Avinash Sharma},
year={2025},
url={https://github.com/paragsarvoday/DeadlineDinosaur},
}
```

## Contact

Contact [Parag Sahu](mailto:parag.sahu@iitgn.ac.in) for questions, comments and reporting bugs, or open a GitHub Issue.


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
- [LiteGS]([https://github.com/YouyuChen0207/DashGaussian](https://github.com/MooreThreads/LiteGS))


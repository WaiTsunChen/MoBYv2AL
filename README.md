# MoBYv2AL
The first contrastive learning work for Active Learning.
Link to paper: https://arxiv.org/abs/2301.01531
Please install the required libraries from requirements.txt
Pip install requirements:
```
pip install -r requirements.txt
```
To run MoBYv2AL (also Random and CoreSet) on CIFAR-10 (also CIFAR-100/SVHN/FashionMNIST): 
```
python main_ssl.py -d cifar10 -m mobyv2al
```
If you find this code useful please cite:

## Citation

```bibtex
@booktitle{caramalau2022mobyv2al,
  title={MoBYv2AL: Self-supervised Active Learning for Image Classification},
  author={Caramalau, Razvan and Bhatarrai, Binod and Stoyanov, Dan and Kim, Tae-Kyun},
  booktitle={BMVC (British Machine Vision Conference)},
  year={2022}
}
```
## Repo structure
For incorporating the Snapshot Serengeti dataset and model training, we adapted the "*.py" files. We added two folders with "_plots" ending. There, our major analysis and plot creations can be found. 

- *reducce_save_umage_size.py* includes the script for preprocessesing and storing the images.
- *sampling_probability.ipynb* includes the analysis for the creation of subset 250'000 samples.
- *temporary.ipynb* is a temporary code dump notebook, where we ended up creating plots for the train/test split, bounding box example and Coreset explanation.

All our model results and source images are stored on the FHNW server, see below. 

## SLURM Example

```
#!/bin/sh

#SBATCH --partition=performance
#SBATCH --time=7-00:00:00

#SBATCH --gres=gpu:1  # never more than 3
#SBATCH --exclude=node22
#SBATCH --exclude=sdas2
#SBATCH --nodelist=gpu23d

singularity exec --nv -B /mnt/nas05/data01/waitsun/snapshotSerengetiCropedResized:/mnt/nas05/data01/waitsun/snapshotSerengetiCropedResized,${HOME}/MoBYv2AL:${HOME}/MoBYv2AL mobyv2al.sif \
  python3 ${HOME}/MoBYv2AL/main_ssl.py -d SnapshotSerengetiSmall -m mobyv2al -c 10 -r 1 -e 222 -la resnet18 -b 128 -ims 64 -id 20240740 -sst coreset -ss 250000 -asi yes
```
For the complete parser argumentation description, have a look at the "main_ssl.py" file.

## Data Storage Information
In case you have permisson on the FHNW Server to my folder, the following is an explanation where to find what files.

Description:
- "result_data_analysis": folder containing model runs. Every run is in a folder. Every folder is constructed the following {dataset_name + method_name + run_id}. The run_id is unique (mostly the running day used).
    
    - "SnapshotSerengetiSmall_mobyv2al_20240533": folder containing all byproducts of a training. If cycle is 10 -> 10 models saved.

- "snapshotSerengeti": folder containing the raw data.

- "snapshoSerengetiCropedResized": folder containing only the cropped and resized animal image as .JPG-file. 
    
    - "df_balanced_top_10_category_lut.df": dataframe with species name corresponding to "df_balanced_top_metadata_test.df" & df_balanced_top_metadata_train.df".

    - "df_balanced_top_10_test.df": dataframe containing the path to the raw images with labels for the test set. Only contains the top 10 most occurring species.

    - "df_balanced_top_10_train.df": datafame containing the path to the raw image with labels for the train set. Only contains the top 10 most occurring species, fully balanced.

    - "df_category_lut_adapted.df": dataframe with species name corresponding to "df_metadata_test.df" & "df_metadata_test.df".

    - "df_metadata_test.df": dataframe containing the path to all raw images with labels for the test set.

    - "df_metadata_train.df": dataframe containing the path to all raw images with labels for the train set, except specie ID 46, since there is no 46 in the test set.

    - "df_metadata_unique_train.df": dataframe containing the path to all raw images with labels for the train set. Only contains one cropped image per image-file.

    - "one_index_per_class_category.npy": numpy array containing predefined data such that each specie appears in the labeled set (use for an applied strategy).
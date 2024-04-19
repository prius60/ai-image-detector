# A detector for AI-generated images
## To start
1. Clone the repo to your workspace.
2. Make sure all dependencies outlined in ***requirements.txt*** is installed in your Python environment
3. Download the [datasets](https://drive.google.com/drive/folders/13ogmgMWfxUNMzXcIWbqusyoodjCjWCQz?usp=sharing) and extract them to the root folder of the repo

## To train the baseline models
```bash
python baseline_model_train.py
```
Examples to evaluate the models are outlined at the bottom of ***baseline_model_train.py***

## To train our final models
```bash
python ViT_Res_patch_train.py
```
By default, we train with extracting 9 random patches. To change the number of patches, add ```--num_patches``` to specify
Examples to evaluate the models are outlined at the bottom of ***ViT_Res_patch_train.py***

## Methods to evaluate each model
```eval.py``` contains helper methods to evaluate our models.

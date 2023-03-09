# Luna16_Monai_Model_XAI_Project




# Building the Docker enviroment

**1.**  Creating the docker container 
```ruby  
docker run --ipc=host --gpus '"device=7"' -it --name=ft42_nnunet_g7 -v /home/ft42/:/Sharedfolder:z -v /local/:/local:z -v /image_data/:/image_data:z -v /image_data2/:/image_data2:z -v /data2/:/data2:z -v /data/:/data:z -v /ssd0/:/ssd0:z -v /net_data/:/net_data:z nvcr.io/nvidia/pytorch:21.12-py3

``` 
**2.**  Installing required libraries
```ruby
pip install nnunet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer

# Install SimpleITK this version otherwise, orthogonality problem will occur and a few cases won't be predicted
python -m pip install SimpleITK==2.0.2
pip install monai
pip install -r requirements.txt

```


# How to Run
run the docker container 
```ruby
docker exec it ft42_nnunet_g7 bash 
```

## Model had Train/Test  at capri, so most of the code base is at @plp-capri.duke.duke and paths corresponds to that**
```ruby
* Code base: /data/usr/ft42/CVIT_XAI/MONAI/detection/
* Data base: /data/usr/ft42/nobackup/LUNGS_XAI_DATA/
```
## Preparing the Environment file and data preparation

**1.**  need to download the Luna16Datasplit files, for our case it's been doanloed and can be accessed at:
```ruby
plp-capri.dhe.duke.edu: /data/usr/ft42/nobackup/LUNGS_XAI_DATA/LUNA16_datasplit
```
This split file containes the 10 predified foldes proposed by Luna16 Challange, annotations and Labels

**2.**  Now need to edit the the environment file amd give the appropriate path based on your systems amd directories.
```ruby
Files: plp-capri.dhe.duke.edu:/data/usr/ft42/CVIT_XAI/MONAI/detection/config/environment_luna16_prepare.json
```
edit the file paths based on your dictecrories:
```ruby
{
    "orig_data_base_dir": "/data/usr/ft42/nobackup/LUNGS_XAI_DATA/LIDC-IDRI",
    "data_base_dir": "/data/usr/ft42/nobackup/LUNGS_XAI_DATA/LIDC-IDRI_resample",
    "data_list_file_path": "/data/usr/ft42/nobackup/LUNGS_XAI_DATA/LUNA16_datasplit/mhd_original/dataset_fold0.json"
}
```

**Required Edits: change the value of "raw_data_base_dir" and "resampled_data_base_dir" to the directory where you store the downloaded images, and the value of "downloaded_datasplit_dir" to where you downloaded the data split json files. [ref:https://github.com/Project-MONAI/tutorials/tree/main/detection]**

**3.**  Pre-processing the Data
```ruby
python3 luna16_prepare_env_files.py
python3 luna16_prepare_images.py -c ./config/config_train_luna16_16g.json
```


## To Train the models (10 folds)

**4.**  Config the Model Traing Hyper-perameters
```ruby
Files: plp-capri.dhe.duke.edu:/data/usr/ft42/CVIT_XAI/MONAI/detection/config/config_train_luna16_16g.json
```
```ruby
{
	"gt_box_mode": "cccwhd",
	"lr": 1e-2,
	"spacing": [0.703125, 0.703125, 1.25],
	"batch_size": 4,
	"patch_size": [192,192,80],
	"val_patch_size": [512,512,208],
	"fg_labels": [0],
	"n_input_channels": 1,
	"spatial_dims": 3,
	"score_thresh": 0.02,
	"nms_thresh": 0.22,
	"returned_layers": [1,2],
	"conv1_t_stride": [2,2,1],
	"base_anchor_shapes": [[6,8,4],[8,6,5],[10,10,6]],
	"balanced_sampler_pos_fraction": 0.3
}

```


**5.** Config Environment file for training a fold, e.g: here fold0:
```ruby
Files: plp-capri.dhe.duke.edu:/data/usr/ft42/CVIT_XAI/MONAI/detection/config/environment_luna16_fold0.json
```
```ruby
{
    "model_path": "/data/usr/ft42/CVIT_XAI/MONAI/detection/base_models/trained_models/model_luna16_fold0.pt",
    "data_base_dir": "/data/usr/ft42/nobackup/LUNGS_XAI_DATA/LIDC-IDRI_resample",
    "data_list_file_path": "/data/usr/ft42/nobackup/LUNGS_XAI_DATA/LUNA16_datasplit/dataset_fold0.json",
    "tfevent_path": "/data/usr/ft42/CVIT_XAI/MONAI/detection/base_models/tfevent_train/luna16_fold0",
    "result_list_file_path": "/data/usr/ft42/CVIT_XAI/MONAI/detection/base_models/result/result_luna16_fold0.json"
}
```
**6.** Run the Fold0 training
```ruby
python3 luna16_training.py -e ./config/environment_luna16_fold0.json -c ./config/config_train_luna16_16g.json
```

**7.** Run the Fold0 testing
```ruby
python3 luna16_testing.py -e ./config/environment_luna16_fold0.json -c ./config/config_train_luna16_16g.json
```


## Evaluation

To perform the evaluation please follow this ref repo: https://github.com/Project-MONAI/tutorials/tree/main/detection#:~:text=3.4%20LUNA16%20Detection%20Evaluation


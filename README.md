# Deep learning technique for susceptibility artefact correction in reversed phase-encoding EPI images
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official code of [S-Net](https://www.sciencedirect.com/science/article/abs/pii/S0730725X19307325?via%3Dihub) and [TS-Net](https://www.mdpi.com/1424-8220/21/7/2314) models for correcting susceptibility artefacts in reversed phase-encoding EPI images.


## Quick start
### Install
1. Install PyTorch=1.5.1 following the [official instructions](https://pytorch.org/)
2. git clone https://github.com/soanduong/S-Net-and-TS-Net-pytorch
3. Install dependencies: pip install -r requirements.txt

### Prepare data
1. Download an example reversed-PE DWI pair [here](https://drive.google.com/drive/folders/1lXnTISmq2cwO5mVHXG9iKZCOiDhwju51?usp=sharing).
2. Save the image pair in your folder, e.g. `data`.
3. Prepare the text file for training and testing
   Each line displays filename of a scan pair. A comma ", " is used to separate the two reversed-PE scans, for example:
   
````
    <filename_forward_PE_scan_pair_1>, <filename_forward_PE_scan_pair_1>, <filename_T1w_1> 
    
    <filename_forward_PE_scan_pair_2>, <filename_forward_PE_scan_pair_2>, <filename_T1w_2>
    ...
    <filename_forward_PE_scan_pair_n>, <filename_forward_PE_scan_pair_n>, <filename_T1w_n>
````
See an example of the text file [here](https://github.com/soanduong/S-Net-and-TS-Net-pytorch/tree/main/data/file_list.txt).
Note that the anatomy T1w images are required for training the TS-Net model.

### Train a model
If you want to train and evaluate our models on your data.
1. Specify the configuration file (refer to configs/snet_base.yml for a reference). Important fields are:
    - dataset_dir: directory of the uncorrected images
    - train_txtfile: file path of the list of image filenames for training
    - test_txtfile: file path of the list of image filenames for training
    - pretrained_model: file path of the pretrained model
    - batch_size: batch size in training the S-Net model
    - n_epochs: number of epochs used in training
    - early_stop: number of epochs to early stop training
                  (loss is not decreased within early_stop epochs)
    - n_displ_fields: dimensions of the displacement field.
                      1 means only the displacement in the phase-encoding direction is estimated.
                      3 means displacements in all three dimensions are estimated.
   
2. Run the training script
For example, train the S-Net on our sample dataset with batchsize of # on # GPUs:
````bash
python iSAC_train.py --config configs/snet_base.yml
````

### Apply the trained model
For example, applying the trained model on the data in data/example_fmri with ...:
````bash
python iSAC_apply.py --config configs/snet_base.yml
                         --trained_model experiments/example_exp/best_model.pth
                         --txtfile data/test.txt
                         --root_dir data/example_fmri
                         --output_prefix iSAC
                         --des_dir data/example_fmri
````

## Citation
If you find this work or code is helpful in your research, please cite our paper:

* [S-Net](https://www.sciencedirect.com/science/article/abs/pii/S0730725X19307325?via%3Dihub): Duong, S.T.M.; Phung, S.L.; Bouzerdoum, A.; Schira, M.M. An unsupervised deep learning technique for susceptibility artifact correction in reversed phase-encoding EPI mages. Magn. Reson. Imaging 2020, 71, 1???10.

* [TS-Net](https://www.mdpi.com/1424-8220/21/7/2314): Duong, S.T.M.; Phung, S.L.; Bouzerdoum, A.; Ang, S.P.; Schira, M.M. Correcting Susceptibility Artifacts of MRI Sensors in Brain Scanning: A 3D Anatomy-Guided Deep Learning Approach. Sensors 2021, 21, 2314, 1-16.

Bibtex
````
@article{Duong2020b,
   author = {Duong, S. T. M. and Phung, S. L. and Bouzerdoum, A. and Schira, M. M.},
   title = {An unsupervised deep learning technique for susceptibility artifact correction in reversed phase-encoding {EPI} mages},
   journal = {Magn. Reson. Imaging},
   volume = {71},
   pages = {1-10},
   year = {2020},
   type = {Journal Article}
}

@article{Duong2020b,
   author = {Duong, S. T. M. and Phung, S. L. and Bouzerdoum, A. and Ang, S. P. and Schira, M. M.},
   title = {Correcting susceptibility artifacts of MRI sensors in brain scanning: a {3D} anatomy-guided deep learning approach,
   journal = {Sensors},
   volume = {21},
   pages = {1-16},
   year = {2021},
   type = {Journal Article}
}
````

## Acknowledgement
* [voxelmorph](https://github.com/voxelmorph/voxelmorph): Learning-based image registration.

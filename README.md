# Deep learning technique for susceptibility artefact correction in reversed phase-encoding EPI images
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/TAMU-VITA/FasterSeg.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/TAMU-VITA/FasterSeg/context:python) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official code of [S-Net](https://www.sciencedirect.com/science/article/abs/pii/S0730725X19307325?via%3Dihub) and [TS-Net](https://www.mdpi.com/1424-8220/21/7/2314) models for correcting susceptibility artefacts in reversed phase-encoding EPI images.


## Quick start
### Install
1. Install PyTorch=0.4.1 following the [official instructions](https://pytorch.org/)
2. git clone https://github.com/soanduong/S-Net-and-TS-Net-pytorch
3. Install dependencies: pip install -r requirements.txt

### Train a model
If you want to train and evaluate our models on your data.
1. Data preparation
2. Please specify the configuration file (refer to configs/snet_base.yml for a reference). Important fields are:
    - dataset_dir: directory of the uncorrected images in txtfiles
    - train_txtfile: file path of the text file that contains filenames of uncorrected image pairs for training
    - test_txtfile: file path of the text file that contains filenames of uncorrected image pairs for testing
    - pretrained_model: file path of the pretrained model
    - batch_size: batch size in training the S-Net model
    - n_epochs: number of epochs used in training
    - early_stop: number of epochs to early stop training
                  (loss is not decreased within early_stop epochs)
    - n_displ_fields: dimensions of the displacement field.
                      1 means only the displacement in the phase-encoding direction is estimated.
                      3 means displacements in all three dimensions are estimated.
   
3. Run the training script
For example, train the S-Net on our sample dataset with batchsize of # on # GPUs:
````bash
python iSAC_train.py --configs configs/snet_base.yml
````

4. Apply the trained model
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
If you find this work or code is helpful in your research, please cite:
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
   title = {Correcting susceptibility artifacts of MRI sensors in brain scanning: a 3D anatomy-guided deep learning approach,
   journal = {Sensors},
   volume = {21},
   pages = {1-16},
   year = {2021},
   type = {Journal Article}
}
````

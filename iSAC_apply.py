# -*- coding: utf-8 -*-
"""
Created on 17/09/2020 5:51 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import os
import yaml
import argparse
import numpy as np
from numpy import loadtxt

# Third party imports
import torch
from tqdm import tqdm
import nibabel as nib

# Local application imports
from models.unet import SNet
from utils.datasets import SACDataset
from utils.general_utils import prepare_device, show_visual_results


def save_nifti(y, ref_file, des_dir, output_prefix):
    """
    Save data y into a nifti file with the reference nifti as ref_file.
    :param y: 3D array
    :param ref_file: full path of a referenced nifti.
    :param des_dir: output directory of the saved file.
    :param output_prefix: prefix for the save nifti.
    :return:
    """
    out_file = '%s/%s_%s' % (des_dir, output_prefix, os.path.basename(ref_file))
    vol_nii = nib.load(ref_file)
    img = nib.Nifti1Image(y, vol_nii.affine)
    nib.save(img, out_file)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Main file')
    args.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--trained_model', default='experiments/snet-hcp-20201012-010138/best_model.pth', type=str,
                      help='File path of the trained model')
    args.add_argument('--txtfile', default='data/test.txt', type=str,
                      help='Text file containing reversed-image pairs.')
    args.add_argument('--root_dir', default='data/example_dwi', type=str,
                      help='Directory to the images in the txtfile.')
    args.add_argument('--output_prefix', default='sac3', type=str,
                      help='Prefix of the output images')
    args.add_argument('--des_dir', default=None, type=str,
                      help='Directory to save the corrected image. Default as the roor_dir')
    args.add_argument('--show_visual', default=False, type=bool,
                      help='Show the subplots of uncorrected and corrected images.')
    cmd_args = args.parse_args()

    assert cmd_args.config is not None, "Please specify a config file"
    if cmd_args.des_dir is None:
        cmd_args.des_dir = cmd_args.root_dir

    # Read experiment configurations
    nn_config_path = cmd_args.config
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Set random seeds for reproducibility
    # init_seeds(cfg['seed'])

    # Use GPU is available, otherwise use CPU
    device, n_gpu_ids = prepare_device()

    # Create an instance of SNet
    m_params = cfg['model_params']
    model = SNet(n_dims=m_params['n_dims'],
                 n_dsp_fields=m_params['n_displ_fields'],
                 nf_enc=m_params['nf_enc'],
                 nf_dec=m_params['nf_dec'],
                 do_batchnorm=m_params['do_batchnorm'], max_norm_val=None)

    # Load the trained model
    print('Loading the trained model: ', cmd_args.trained_model)
    model.load_state_dict(torch.load(cmd_args.trained_model, map_location='cpu'))
    model = model.to(device)

    # Get image pairs from the text file
    img_files = loadtxt(cmd_args.txtfile, delimiter=', ', dtype=np.str)
    # Implement the network for each pair of image

    # Configuring test dataloader
    test_dataset = SACDataset(cmd_args.root_dir, cmd_args.txtfile)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  pin_memory=True, num_workers=0)
                                                  #num_workers=cfg['num_workers'])

    model.eval()  # set model to eval mode
    pbar = tqdm(test_dataloader, ncols=80, desc='Testing')
    running_performance = 0
    with torch.no_grad():  # declare no gradient operations, and namespacing in comet
        for step, minibatch in enumerate(pbar):
            # Test data
            x_fwd, x_inv = minibatch['fwd_img'], minibatch['inv_img']
            file_fwd, file_inv = minibatch['fwd_file'], minibatch['inv_file']

            x_fwd = x_fwd.to(device)
            x_inv = x_inv.to(device)

            # Forward pass
            y_fwd, y_inv, dsp_fields = model(x_fwd, x_inv)
            y_fwd = y_fwd.detach().cpu().numpy()
            y_inv = y_inv.detach().cpu().numpy()
            dsp_fields = dsp_fields.detach().cpu().numpy()

            # Show visual results
            if cmd_args.show_visual:
                show_visual_results(x_fwd.detach().cpu().numpy(),
                                    x_inv.detach().cpu().numpy(),
                                    y_fwd, y_inv, dsp_fields,
                                    show_visual=True)

            # Save the output corrected forward image
            save_nifti(y_fwd[0, 0, ...], file_fwd[0],
                       cmd_args.des_dir, cmd_args.output_prefix)

            # Save the output corrected inverse image
            save_nifti(y_inv[0, 0, ...], file_inv[0],
                       cmd_args.des_dir, cmd_args.output_prefix)

    print('Done')
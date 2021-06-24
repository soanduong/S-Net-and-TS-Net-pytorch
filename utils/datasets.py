# -*- coding: utf-8 -*-
"""
Created on 24/06/2020 1:26 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


class SACDataset(Dataset):
    def __init__(self, root_dir, txtfile):
        """
        :param root_dir: the root directory to the dataset folder, e.g '/data/DLSAC/data/HCP_dataset/'
        :param txtfile: a file that contains the path of image pair.
                        Each line the text file is constructed as:
                        path_of_fwd_img, path_of_inv_img, pe_dir, path_of_anat
        """
        super(SACDataset, self).__init__()
        self.training_file = txtfile
        self.root_dir = root_dir
        self.scan_pairs = np.loadtxt(txtfile, dtype=np.str, comments="#", delimiter=", ", unpack=False)

        # Check if having only one image pair
        if isinstance(self.scan_pairs[0], str):
            self.scan_pairs = [self.scan_pairs]
        # print('root dir:', self.root_dir)

    def __len__(self):
        """
        :return: the size of the dataset, i.e. the number of image pairs
        """
        return len(self.scan_pairs)

    def __getitem__(self, index):
        scan = self.scan_pairs[index]
        # print(self.root_dir)
        # print(len(scan), scan)
        fwd_img_file = '%s/%s' % (self.root_dir, scan[0].strip())
        inv_img_file = '%s/%s' % (self.root_dir, scan[1].strip())
        # print('fwd. img:', fwd_img_file)
        # print('inv. img:', inv_img_file)

        # Read the scan pair
        X_fwd = np.float32(nib.load(fwd_img_file).get_data())[np.newaxis, ...]
        X_inv = np.float32(nib.load(inv_img_file).get_data())[np.newaxis, ...]

        # # Convert the PE direction if necessary

        # if pe_dir != 0:
        #     X_fwd = permute_data(X_fwd, pe_dir)
        #     X_inv = permute_data(X_inv, pe_dir)

        # convert the images into Pytorch tensors
        X_fwd = torch.Tensor(X_fwd)
        X_inv = torch.Tensor(X_inv)

        # Read the anat image if it is possible
        if len(scan) > 2:
            X_anat = np.float32(nib.load(self.root_dir + scan[-1].strip()).get_data())[np.newaxis, ...]
            X_anat = torch.Tensor(X_anat)
            return {'fwd_img': X_fwd, 'inv_img': X_inv, 'anat_img': X_anat,
                    'fwd_file': fwd_img_file, 'inv_file': inv_img_file}

        return {'fwd_img': X_fwd, 'inv_img': X_inv,
                'fwd_file': fwd_img_file, 'inv_file': inv_img_file}


class SNetDataset(Dataset):
    def __init__(self, root_dir, training_file):
        """
        :param root_dir: the root directory to the dataset folder, e.g '/data/DLSAC/data/HCP_dataset/'
        :param training_file: a file that contains the path of image pair.
                              Each line the text file is constructed as:
                              path_of_fwd_img; path_of_inv_img; pe_dir; path_of_anat
        """
        super(SNetDataset, self).__init__()
        self.training_file = training_file
        self.root_dir = root_dir
        self.scan_pairs = np.loadtxt(training_file, dtype=np.str, comments="#", delimiter=";", unpack=False)

    def __len__(self):
        """
        :return: the size of the dataset, i.e. the number of image pairs
        """
        return len(self.scan_pairs)

    def __getitem__(self, index):
        from utils.general_utils import permute_data  # to fix circular dependency

        scan = self.scan_pairs[index]

        # read the scan pair
        X_fwd = np.float32(nib.load(self.root_dir + scan[0].strip()).get_data())[np.newaxis, ...]
        X_inv = np.float32(nib.load(self.root_dir + scan[1].strip()).get_data())[np.newaxis, ...]

        # Convert the PE direction if necessary
        pe_dir = np.int(scan[2].strip())
        if pe_dir != 0:
            X_fwd = permute_data(X_fwd, pe_dir)
            X_inv = permute_data(X_inv, pe_dir)

        # convert the images into Pytorch tensors
        X_fwd = torch.Tensor(X_fwd)
        X_inv = torch.Tensor(X_inv)

        return {'fwd_img': X_fwd, 'inv_img': X_inv, 'pe_dir': pe_dir}


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------
# def snet_dataset_torchio(txtfile, transforms=None, patch_based=False, num_workers=1):
#     """
#     This function define the dataset function for training the SNet module.
#     The defined dataset is based on the torchio (https://arxiv.org/abs/2003.04696)
#
#     :param txtfile: a file that contains the path of image pair.
#                           Each line the text file is constructed as:
#                           path_of_fwd_img; path_of_inv_img; pe_dir; path_of_anat
#     :param transforms:
#     :param patch_based:
#     :param num_workers:
#     :return:
#     """
#     # get the (fwd_path, inv_path) pairs from the text file
#     scan_pairs = loadtxt(txtfile, dtype=np.str, comments="#", delimiter=";", unpack=False)
#
#     # create a list of subject, each subject contains a pair of scan
#     subjects = []
#     for scan in scan_pairs:
#         subject = torchio.Subject({'fwd_img': torchio.Image(scan[0].strip(), torchio.INTENSITY),
#                                    'inv_img': torchio.Image(scan[1].strip(), torchio.INTENSITY),
#                                    'pe_dir': torchio.LABEL(np.int(scan[2].strip()))})
#         subjects.append(subject)
#     dataset = torchio.ImagesDataset(subjects, transform=transforms)
#     if patch_based:
#         patch_size = 32
#         samples_per_volume = 12
#         max_queue_length = 300
#         patch_dataset = torchio.Queue(
#             subjects_dataset=dataset,
#             max_length=max_queue_length,
#             samples_per_volume=samples_per_volume,
#             patch_size=patch_size,
#             sampler_class=torchio.sampler.ImageSampler,
#             num_workers=num_workers,
#             shuffle_subjects=True,
#             shuffle_patches=True,
#         )
#         return patch_dataset
#     else:
#         return dataset


if __name__ == "__main__":
    dataset_name = 'hcp3Tfmri'
    training_file = 'data/datasets/%s_train.dat' % dataset_name
    ubuntu_dir = '/data/DLSAC/data/'
    pc_dir = 'Z:/projects/DLSAC/'
    scan_pairs = np.loadtxt(training_file, dtype=np.str, comments="#", delimiter=";", unpack=False)

    scan = scan_pairs[1]
    pe_dir = np.int(scan[2].strip())

    # read the scan pair
    X_fwd = np.float32(nib.load(scan[0].replace(ubuntu_dir, pc_dir).strip()).get_data())[np.newaxis, np.newaxis, ...]
    X_inv = np.float32(nib.load(scan[1].replace(ubuntu_dir, pc_dir).strip()).get_data())[np.newaxis, np.newaxis, ...]

    # convert the images into Tensors
    X_fwd = torch.Tensor(X_fwd)
    X_inv = torch.Tensor(X_inv)

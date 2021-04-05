import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.datasets import SNetDataset, SACDataset


def prepare_device(n_gpu_use=1):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def init_obj(module_name, module_args, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    assert all([k not in module_args for k in
                kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def permute_data(x, pe_dir):
    """
    This function permutes data of the tensor x of size (batch, n_channels, 1st_size, ..., nth_size)
    so that the PE direction become the first dimension (except the batch and n_channels dimensions).
    :param x: the tensor of size (batch, n_channels, 1st_size, ..., nth_size)
    :param pe_dir: the number indicating the PE direction of the image of size (1st_size, ..., nth_size)
    :return: a tensor of size (batch, n_channels, pe_direction_th_size, ..., 1st_size, ..., nth_size)
    """
    # set the permutation direction so that the pe_dir and the first directions are swapped
    per_seq = list(range(len(x.shape) - 2))
    per_seq[0] = pe_dir
    per_seq[pe_dir] = 0
    per_seq = [k + 2 for k in per_seq]

    # permute the data
    if len(x.shape) - 2 == 2:  # for 2D images
        x = x.permute(0, 1, per_seq[0], per_seq[1])
    elif len(x.shape) - 2 == 3:  # for 3D images
        x = x.permute(0, 1, per_seq[0], per_seq[1], per_seq[2])
    return x


def get_sac_dataloaders(cfg):
    '''
    Return train, validation and test dataloaders for the experiment.
    :param cfg: a dict containing all the required settings for the dataloaders
    :return:
    '''

    # Configuring train and val dataloaders
    #print('root dir:', cfg['dataset_dir'])
    train_dataset = SACDataset(cfg['dataset_dir'], cfg['train_txtfile'])

    # 0.9 : 0.1 split for train:val subsets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.9 * num_train))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=cfg['num_workers']
    )

    val_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=cfg['num_workers']
    )

    # Configuring test dataloader
    test_dataset = SACDataset(cfg['dataset_dir'], cfg['test_txtfile'])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # restrict the batch_size to 1 because we may need to convert the direction of some data.
        pin_memory=True, num_workers=cfg['num_workers']
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_experiment_dataloaders(cfg):
    '''
    Return train, validation and test dataloaders for the experiment.
    :param cfg: a dict containing all the required settings for the dataloaders
    :return:
    '''

    # Configuring train and val dataloaders
    train_dataset = SNetDataset(cfg['dataset_dir'], cfg['train_txtfile'])

    # 0.9 : 0.1 split for train:val subsets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.9 * num_train))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]),
        pin_memory=True, num_workers=cfg['num_workers']
    )

    val_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True, num_workers=cfg['num_workers']
    )

    # Configuring test dataloader
    test_dataset = SNetDataset(cfg['dataset_dir'], cfg['test_txtfile'])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # restrict the batch_size to 1 because we may need to convert the direction of some data.
        pin_memory=True, num_workers=cfg['num_workers']
    )

    return train_dataloader, val_dataloader, test_dataloader


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.mkdir(path)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))


def init_seeds(seed):
    # Setting seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def show_visual_results(x_fwd, x_inv, y_fwd, y_inv, disp_fields, show_visual=0,
                        comet=None, fig_name=""):
    """
    This function shows the results of input and output images and the displacement fields.
    :param x_fwd: an array of size (batch, 1, H, W, D)
    :param x_inv: an array of size (batch, 1, H, W, D)
    :param y_fwd: an array of size (batch, 1, H, W, D)
    :param y_inv: an array of size (batch, 1, H, W, D)
    :param disp_fields: an array of size (batch, 3, H, W, D)
    :param show_visual: a boolean to display the figure or not
    :param comet: the comet logger object
    :param fig_name: a string as the figure name for the comet logger
    :return:
    """

    h = 2
    if plt.fignum_exists(h):  # close if the figure existed
        plt.close()
    fig = plt.gcf()

    # show slices
    sl = np.int32(y_fwd.shape[4] / 2)
    plt.figure(1)
    plt.suptitle('Slice = %d' % sl)
    # uncorrected images
    plt.subplot(331)
    plt.imshow(np.rot90(x_fwd[0, 0, :, :, sl]))
    plt.title('Input fwd image')
    plt.subplot(332)
    plt.imshow(np.rot90(x_inv[0, 0, :, :, sl]))
    plt.title('Input inv image')
    plt.subplot(333)
    plt.imshow(np.rot90(x_fwd[0, 0, :, :, sl] - x_inv[0, 0, :, :, sl]))
    plt.title('Input diff.')

    # corrected images
    plt.subplot(334)
    plt.imshow(np.rot90(y_fwd[0, 0, :, :, sl]))
    plt.title('Output fwd image')
    plt.subplot(335)
    plt.imshow(np.rot90(y_inv[0, 0, :, :, sl]))
    plt.title('Output inv image')
    plt.subplot(336)
    plt.imshow(np.rot90(y_fwd[0, 0, :, :, sl] - y_inv[0, 0, :, :, sl]))
    plt.title('Output diff.')

    # displacement fields
    plt.subplot(337)
    plt.imshow(np.rot90(disp_fields[0, 0, :, :, sl]))
    plt.title('1st displ. field')
    plt.subplot(338)
    plt.imshow(np.rot90(disp_fields[0, 1, :, :, sl]))
    plt.title('2nd displ. field')
    plt.subplot(339)
    plt.imshow(np.rot90(disp_fields[0, 2, :, :, sl]))
    plt.title('3rd displ. field')

    if show_visual:
        plt.show()

    if comet is not None:
        comet.log_figure(figure_name=fig_name, figure=fig)


if __name__ == '__main__':
    print(prepare_device(n_gpu_use=1))

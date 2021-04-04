# -*- coding: utf-8 -*-
"""
Created on 17/09/2020 5:51 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import time
from comet_ml import Experiment
import os
import yaml
import shutil
import argparse
import numpy as np

# Third party imports
import torch
from tqdm import tqdm

# Local application imports
import utils.losses
from models.unet import SNet
from utils.general_utils import prepare_device, init_obj, \
    get_sac_dataloaders, create_exp_dir, show_visual_results


def main(cfg, comet):
    '''
    :param cfg: the parameters specified in the .yml config file
    '''

    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])

    # Use GPU is available, otherwise use CPU
    device, n_gpu_ids = prepare_device()

    # Get datalodaers
    train_loader, val_loader, test_loader = get_sac_dataloaders(cfg['train_params'])

    # Create the model
    m_params = cfg['model_params']
    model = SNet(n_dims=m_params['n_dims'],
                 n_dsp_fields=m_params['n_displ_fields'],
                 nf_enc=m_params['nf_enc'],
                 nf_dec=m_params['nf_dec'],
                 do_batchnorm=m_params['do_batchnorm'], max_norm_val=None)
    # summary(model, [(1, 90, 104, 72), (1, 90, 104, 72)], device=device.type)

    # Load the pretrained model if it is required
    if os.path.isfile(cfg['train_params']['pretrained_model']):
        model.load_state_dict(torch.load(cfg['train_params']['pretrained_model'],
                                         map_location='cpu'))
    model = model.to(device)

    # Define an optimiser
    print(f"Optimiser for model weights: {cfg['optimizer']['type']}")
    optimizer = init_obj(cfg['optimizer']['type'],
                         cfg['optimizer']['args'],
                         torch.optim, model.parameters())

    # Metrics
    metric = getattr(utils.losses, cfg['sim_loss']['type'])

    # Loss function
    loss_fn = utils.losses.SNetLoss(cfg['sim_loss']['weights'])

    # Training
    training(cfg['train_params'], optimizer, model, train_loader, val_loader,
             loss_fn, metric, device, comet)

    # Testing
    print("\nThe training has completed. Testing the model now...")

    # Load the best model
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    model.load_state_dict(saved)
    testing(model, test_loader, metric, device, comet)

    comet.log_asset(cfg['train_params']['save_dir'] + '/best_model.pth')


def training(train_cfg, optimizer, model, train_loader, val_loader, loss_fn,
             metric, device, comet=None):
    '''
    :param train_cfg: the training configuration dict
    :param optimizer: the optimizer
    :param model:  the model
    :param train_loader:  the train dataloader
    :param val_loader:  the validation dataloader
    :param loss_fn: the loss function
    :param metric:  the metric
    :param device: the training device (cpu or gpu)
    :param comet: comet-ml logger
    :return:
    '''
    epochs = train_cfg['n_epochs']
    best_model_indicator = 10000000
    not_improved_epochs = 0
    for epoch in range(epochs):
        print(f"\nTraining epoch {epoch + 1}/{epochs}")
        print("-----------------------------------")
        # Training
        with comet.train():
            train_loss, train_performance = train_epoch(optimizer, model,
                                                        train_loader, loss_fn,
                                                        metric, device)
            comet.log_metric('loss', train_loss, epoch=epoch + 1)
            comet.log_metric('performance', train_performance, epoch=epoch + 1)

        # Validating
        with comet.validate():
            val_loss, val_performance = val_epoch(model, val_loader, loss_fn,
                                                  metric, device)
            comet.log_metric('loss', train_loss, epoch=epoch + 1)
            comet.log_metric('performance', val_performance, epoch=epoch + 1)

        print(f"\nSummary of epoch {epoch + 1}:")
        print(f"Train loss: {train_loss:.4f}, "
              f"Train performance: {train_performance:.4f}")

        print(f"Val loss: {val_loss:.4f}, "
              f"Val performance: {val_performance:.4f}")

        # Save best model only
        if val_loss < best_model_indicator:
            print(
                f'Model exceeds prev best score'
                f'({val_loss:.4f} < {best_model_indicator:.4f}). Saving it now.')
            best_model_indicator = val_loss
            # Save model
            torch.save(model.state_dict(),
                       train_cfg['save_dir'] + '/best_model.pth')
            not_improved_epochs = 0  # reset counter
        else:
            if not_improved_epochs > train_cfg['early_stop']:  # early stopping
                print(
                    f"Stopping training early because it has not improved for "
                    f"{train_cfg['early_stop']} epochs.")
                break
            else:
                not_improved_epochs = not_improved_epochs + 1


def train_epoch(optimizer, model, train_loader, loss_fn, metric, device):
    '''
    Logic for a training epoch
    :param optimizer: the optimizer for the model
    :param model: the network model
    :param train_loader: the training dataloader
    :param loss_fn: the loss function for the model
    :param metric: the metric for performance evaluation
    :param device: do the training on cpu or gpu
    :return: train loss and train performance for this epoch
    '''

    model.train()
    pbar = tqdm(train_loader, ncols=80, desc='Training')
    running_loss = 0
    running_performance = 0
    for step, minibatch in enumerate(pbar):
        optimizer.zero_grad()  # clear the old gradients
        # Train data
        x_fwd, x_inv = minibatch['fwd_img'], minibatch['inv_img']
        x_fwd = x_fwd.to(device)
        x_inv = x_inv.to(device)

        # Forward pass
        y_fwd, y_inv, dsp_fields = model(x_fwd, x_inv)

        # Compute loss, then update weights
        loss = loss_fn(y_fwd, y_inv, dsp_fields)
        loss.backward()  # calculate gradients
        optimizer.step()  # update weights

        # Evaluate train performance
        performance = metric(y_fwd, y_inv)

        # Storing the loss and metric
        running_loss = running_loss + loss.detach().cpu().numpy()
        running_performance = running_performance + performance.detach().cpu().numpy()

        # Display losses
        result = "{}: {:.4}".format('Train loss', loss)
        pbar.set_postfix_str(result)

    avg_loss = running_loss / len(train_loader)
    avg_performance = running_performance / len(train_loader)

    return avg_loss, avg_performance


def val_epoch(model, val_loader, loss_fn, metric, device):
    model.eval()  # set model to eval mode
    pbar = tqdm(val_loader, ncols=80, desc='Validating')
    running_loss = 0
    running_performance = 0
    with torch.no_grad():  # declare no gradient operations
        for step, minibatch in enumerate(pbar):
            # Val data
            x_fwd, x_inv = minibatch['fwd_img'], minibatch['inv_img']
            x_fwd = x_fwd.to(device)
            x_inv = x_inv.to(device)

            # Forward pass
            y_fwd, y_inv, dsp_fields = model(x_fwd, x_inv)

            # Compute loss
            loss = loss_fn(y_fwd, y_inv, dsp_fields)

            # Evaluate train performance
            performance = metric(y_fwd, y_inv)

            # Storing the loss and metric
            running_loss = running_loss + loss.detach().cpu().numpy()
            running_performance = running_performance + performance.detach().cpu().numpy()

            # Display losses
            result = "{}: {:.4}".format('Val loss', loss)
            pbar.set_postfix_str(result)

        avg_loss = running_loss / len(val_loader)
        avg_performance = running_performance / len(val_loader)

        return avg_loss, avg_performance


def testing(model, test_loader, metric, device, comet):
    model.eval()  # set model to eval mode
    pbar = tqdm(test_loader, ncols=80, desc='Testing')
    running_performance = 0
    with torch.no_grad():  # declare no gradient operations, and namespacing in comet
        for step, minibatch in enumerate(pbar):
            # Test data
            x_fwd, x_inv = minibatch['fwd_img'], minibatch['inv_img']
            x_fwd = x_fwd.to(device)
            x_inv = x_inv.to(device)

            # Forward pass
            y_fwd, y_inv, dsp_fields = model(x_fwd, x_inv)

            # Evaluate train performance
            performance = metric(y_fwd, y_inv)

            # Storing the loss and metric
            running_performance = running_performance + performance.detach().cpu().numpy()

            # Show visual results
            show_visual_results(x_fwd.detach().cpu().numpy(),
                                x_inv.detach().cpu().numpy(),
                                y_fwd.detach().cpu().numpy(),
                                y_inv.detach().cpu().numpy(),
                                dsp_fields.detach().cpu().numpy(),
                                show_visual=False, comet=comet, fig_name=step)

        avg_performance = running_performance / len(test_loader)

    print(f"Testing performance: {avg_performance:.4f}")
    comet.log_metric('test_performance', avg_performance)


def init_seeds(seed):
    # Setting seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Main file')
    args.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--debug', default=0, type=int,
                      help='debug mode? (default: 0')
    args.add_argument('--disable_comet', default=True, type=bool,
                      help='Disable comet in training')
    cmd_args = args.parse_args()

    assert cmd_args.config is not None, "Please specify a config file"

    # Configuring comet-ml logger
    comet_key = ''
    comet_workspace = 'soanduong'
    if not cmd_args.disable_comet:
        api_key_path = "./configs/comet-ml-key.yml"
        if os.path.isfile(api_key_path) and os.access(api_key_path, os.R_OK):
            with open(api_key_path) as file:
                comet_cfg = yaml.load(file, Loader=yaml.FullLoader)
                comet_key = comet_cfg['comet_key']
                comet_workspace = comet_cfg['comet_workspace']
        else:
            raise FileNotFoundError(
                'You need to create a yaml containing 2 fields: '
                '                  + comet_key: comet-ml api key, '
                '                  + comet_workspace: workspace of the comet. ' 
                'The full path should be ./configs/comet-ml-key.yml')

    disable_comet = bool(cmd_args.debug) and cmd_args.disable_comet
    comet = Experiment(api_key=comet_key,
                       project_name="iSAC",
                       workspace=comet_workspace,
                       disabled=disable_comet,
                       auto_metric_logging=False)

    # Read experiment configurations
    nn_config_path = cmd_args.config
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    if cmd_args.debug == 1:
        cfg['num_workers'] = 0
        cfg['warmup_epochs'] = 1
        print('DEBUG mode')
        save_dir = 'experiments/test-folder'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        # If not debug, we create a folder to store the model weights and etc
        save_dir = f'experiments/{cfg["name"]}-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)

    # Copy the configuration file into the save_dir
    shutil.copy(nn_config_path, save_dir)

    cfg['train_params']['save_dir'] = save_dir
    comet.set_name(cfg['name'])
    comet.log_asset(nn_config_path)
    comet.add_tags(cfg['tags'])

    main(cfg, comet)

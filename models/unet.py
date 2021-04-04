import torch.nn as nn
import torch
import numpy as np
import models
from utils.general_utils import permute_data


class SNet(nn.Module):
    """
    [UNet] is a class representing the U-Net implementation that takes a pair of input images (reversed-PE images)
    and outputs the n displacement fields:
            +) n = 1: the displacement field in the first dimension (PE direction)
            +) n = 3: three displacement fields in all three dimensions.
            +) n = 5: five displacement fields, in which
                    - the 1st displacement field is of the fwd image and in the PE direction.
                    The negative of this displacement field is of the inv image in the PE direction.
                    - the 2nd and 3rd displacement fields are of the fwd image and in the other two dimensions.
                    - the 4th and 5th displacement fields are of the inv image and in the other two dimensions.
    """

    def __init__(self, n_dims=3, n_dsp_fields=3, nf_enc=[16, 32, 32, 32], nf_dec=[32, 32, 32, 32, 8, 8],
                 do_batchnorm=False, max_norm_val=None, interp_mode="bilinear"):
        """
        :param n_dims: number of dimensions of the input data
        :param n_dsp_fields: number of the predicted displacment fields
        :param nf_enc: list of number of filters in the encoder
        :param nf_dec: list of number of filters in the decoder
        :param do_batchnorm: a boolean param indicating whether a BatchNormalization layer is added at
                            the end of the convolutional block (default = False)
        :param max_norm_val: a scalar indicating the maximum value of the intensity normalization
                            (default = None: no intensity normalization)
        :param interp_mode: method of interpolation for grid_sampler
        """
        super(SNet, self).__init__()
        assert n_dims in [2, 3], "Dimensions should be 2 or 3. Found: %d" % n_dims
        assert n_dsp_fields in [1, 3, 5], "Dimension should be 1, 3, or 5. Found: %d" % n_dsp_fields

        self.n_dims = n_dims
        self.n_dsp_fields = n_dsp_fields
        self.nf_enc = nf_enc
        self.nf_dec = nf_dec
        self.do_batchnorm = do_batchnorm
        self.max_norm_val = max_norm_val
        self.interp_mode = interp_mode

        self.enc_dec = UNet(n_dims=n_dims, n_dsp_fields=n_dsp_fields, nf_enc=nf_enc, nf_dec=nf_dec,
                            do_batchnorm=do_batchnorm, max_norm_val=max_norm_val)
        # self.stu = STU3d(mode=interp_mode)
        stu = globals()['STU%dd' % n_dsp_fields]
        self.stu = stu(mode=interp_mode)

    def forward(self, x_fwd, x_inv):
        # concatenate the two input images as an input for the encoder-decoder
        x = torch.cat([x_fwd, x_inv], dim=1)

        # pass the input through the encoder-decoder
        dsp_fields = self.enc_dec(x)

        # unwarp the original image with the predicted displacement fields
        y_fwd, y_inv = self.stu(x_fwd, x_inv, dsp_fields)

        return y_fwd, y_inv, dsp_fields


class UNet(nn.Module):
    """
    [UNet] is a class representing the U-Net implementation that takes a pair of input images (reversed-PE images)
    and outputs the n displacement fields:
            +) n = 1: the displacement field in the first dimension (PE direction)
            +) n = 3: three displacement fields in all three dimensions.
            +) n = 5: five displacement fields, in which
                    - the 1st displacement field is of the fwd image and in the PE direction.
                    The negative of this displacement field is of the inv image in the PE direction.
                    - the 2nd and 3rd displacement fields are of the fwd image and in the other two dimensions.
                    - the 4th and 5th displacement fields are of the inv image and in the other two dimensions.
    """

    def __init__(self, n_dims=3, n_dsp_fields=3, nf_enc=[16, 32, 32, 32], nf_dec=[32, 32, 32, 32, 8, 8],
                 do_batchnorm=False, max_norm_val=None):
        """

        :param n_dims: number of dimensions of the input data
        :param n_dsp_fields: number of the predicted displacment fields
        :param nf_enc: list of number of filters in the encoder
        :param nf_dec: list of number of filters in the decoder
        :param do_batchnorm: a boolean param indicating whether a BatchNormalization layer is added at
                            the end of the convolutional block (default = False)
        :param max_norm_val: a scalar indicating the maximum value of the intensity normalization
                            (default = None: no intensity normalization)
        """
        super(UNet, self).__init__()
        assert n_dims in [2, 3], "Dimensions should be 2 or 3. Found: %d" % n_dims
        assert n_dsp_fields in [1, 3, 5], "Dimension should be 1, 3, or 5. Found: %d" % n_dsp_fields

        self.n_dims = n_dims
        self.n_dsp_fields = n_dsp_fields
        self.nf_enc = nf_enc
        self.nf_dec = nf_dec
        self.do_batchnorm = do_batchnorm
        self.max_norm_val = max_norm_val

        self.norm_input = NormLayer(mode='pad', max_norm_val=max_norm_val, divisible_number=2 ** len(nf_enc))
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        self.recovery_size = NormLayer(mode='crop')

        # define the encoder
        for (idx, n_filters) in enumerate(nf_enc):
            if idx == 0:  # first filter
                c_in = 2
            else:
                c_in = nf_enc[idx - 1]
            self.encoder.append(conv_block(c_in, n_filters, conv_dim=n_dims,
                                           stride=2, do_batchnorm=do_batchnorm))

        # define the decoder
        n_encoder_layers = len(nf_enc)
        n_decoder_layers = len(nf_dec)
        for idx in range(n_encoder_layers):
            if idx == 0:
                c_in = nf_enc[-1]
            else:
                c_in = nf_dec[idx - 1] + nf_enc[n_encoder_layers - idx - 1]
            self.decoder.append(conv_block(c_in, nf_dec[idx], conv_dim=n_dims,
                                           do_upsample=True, do_batchnorm=do_batchnorm))

        # define the post_layer
        for idx in range(n_encoder_layers, n_decoder_layers):
            c_in = nf_dec[idx - 1]
            if idx == n_decoder_layers - 1:
                c_in = c_in + 2
            self.post_layers.append(conv_block(c_in, nf_dec[idx], conv_dim=n_dims,
                                               do_upsample=False, do_batchnorm=do_batchnorm))

        # append a conv_block that output n_dsp_fields channels
        self.post_layers.append(conv_block(nf_dec[-1], n_dsp_fields, conv_dim=n_dims,
                                           do_upsample=False, do_batchnorm=do_batchnorm))

    def forward(self, x):
        """
        Pass input x through the UNet forward once
        :param x: concatenated input image pair, size of (batch, 2, 1st_size, ..., nth_size)
        :return: the tensor representing the displacement field of the input image pair x.
        """
        # normalize the input
        x, padding = self.norm_input(x)
        input_org = x  # will be used by last post_layer via skip-connect

        # add layers in the encoder (E1-4)
        enc_outputs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)  # store encoder output

        # add layers in the decoder (D1-4)
        n_encoder_layers = len(self.nf_enc)
        for (idx, layer) in enumerate(self.decoder):
            # concatenate the output with the corresponding encoder layer if it is required
            if idx != 0:  # concatenate on the channel dimension
                x = torch.cat((x, enc_outputs[n_encoder_layers - idx - 1]), 1)

            # add the decoder layer
            x = layer(x)

        # add layers after the decoder (D5 and D6)
        for (idx, layer) in enumerate(self.post_layers):
            # concatenate the input of second last layer with the original
            if idx == (len(self.post_layers) - 2):
                x = torch.cat((x, input_org), 1)

            # add the layer
            x = layer(x)

        # recovery the size of the input
        x = self.recovery_size(x, padding)

        return x


class NormLayer(nn.Module):
    """
    - [NormLayer] is a class that can pad the input N-D tensor so that the size
     in each dimension can be divisible by a divisible_number (default = 16).

    - This class can also crop the input N-D tensor given a padding size.

    Note that the format of the input data: (batch, n_channels, 1st_size, ..., nth_size)
    """

    def __init__(self, mode="pad", divisible_number=16, max_norm_val=None):
        """
        :param mode: "pad" or "crop"
        """
        super(NormLayer, self).__init__()
        self.mode = mode
        self.divisible_number = divisible_number
        self.max_norm_val = max_norm_val

    def forward(self, x, padding=None):
        # normalize the x into [0, self.max_val]
        if self.max_norm_val is not None:
            print(self.max_norm_val)
            x = norm_intensity(x, max_val=self.max_norm_val)

        if self.mode == "pad":  # do padding
            # compute the padding amount so that each dimension is divisible by divisible_number
            padding = compute_padding(x, divisible_number=self.divisible_number)
            # print('padding', padding)
            # apply padding
            x = nn.functional.pad(x, padding)

            return x, padding

        else:  # do cropping
            # cropping by padding with minus amount of padding
            padding = tuple([-p for p in padding])
            # print('cropping', padding)

            # apply cropping
            x = nn.functional.pad(x, padding)

            return x


class STU3d(nn.Module):
    """
    [STU3d] represents the spatial transformation block that uses the output of
    the encoder-decoder, as the 3 displacement fields in the x, y, and z-axes,
    to perform an grid_sample for a pair of reversed-PE images.

    """

    def __init__(self, mode='bilinear'):
        """
        Instantiate the block
            :param mode: method of interpolation for grid_sampler
        """
        super(STU3d, self).__init__()
        self.mode = mode

    def forward(self, x_fwd, x_inv, dsp_fields):
        """
        Push the (x_fwd, +dsp_fields) and (x_inv, -dsp_fields) through the spatial transform block
        to obtain the unwarpped images
            :param x_fwd: the original forward image with a size of: (batch, 2, 1st_size, ..., nth_size)
            :param x_inv: the original inverse image with a size of: (batch, 2, 1st_size, ..., nth_size)
            :param dsp_fields: the displacement fields (output of the encoder-decoder network)
        """
        # Create sampling grid
        vol_size = dsp_fields.shape[2:]
        vectors = [torch.arange(0, s) for s in vol_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(x_fwd.device)

        # unwarp the x_fwd with the +dsp_field
        # unwarp the x_inv with the -dsp_field
        y_fwd = interp(x_fwd, grid + dsp_fields, vol_size, self.mode)
        y_inv = interp(x_inv, grid - dsp_fields, vol_size, self.mode)

        return y_fwd, y_inv
        # return torch.cat((y_fwd, y_inv), 1)


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------
def conv_block(c_in, c_out, conv_dim=3, stride=1, kernel_size=3, negative_slope=0.2,
               do_upsample=False, do_batchnorm=False):
    """
    Creates a convolutional building block: Conv + LeakyReLU + Upsample (optional) + BatchNorm (optional)

    :param c_in: input channel size
    :param c_out: output channel size
    :param conv_dim: which dim of conv to use (2:2D, 3:3D)
    :param kernel_size: filter size of the conv layer
    :param stride: stride of the convolutional layer
    :param negative_slope: the parameter that controls the angle of the negative slope of the LeakyReLU layer
    :param do_upsample: a boolean param indicating whether an upsample layer is added after the (Conv + LeakyReLU)
    :param do_batchnorm: a boolean param indicating whether an upsample layer is added at the end of the block
    :return: a convolutional building block
    """

    Conv = getattr(nn, 'Conv%dd' % conv_dim)

    block = nn.ModuleList()
    # compute the padding amount
    padding = int(np.ceil(((kernel_size - 1) + 1 - stride) / 2.))

    block.append(Conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding))
    block.append(nn.LeakyReLU(negative_slope))

    # append an Upsample layer if it is required
    if do_upsample:
        if conv_dim == 2:
            upsample_mode = 'bilinear'
        elif conv_dim == 3:
            upsample_mode = 'trilinear'
        block.append(nn.Upsample(scale_factor=2, mode=upsample_mode))

    # append an BatchNormalization layer if it is required
    if do_batchnorm:
        BatchNorm = getattr(nn, 'BatchNorm%dd' % conv_dim)
        block.append(BatchNorm(c_out))

    return nn.Sequential(*block)


def compute_padding(x, divisible_number=16):
    """
    Computes the padding for each spatial dim (exclude depth) so that it is divisible by divisible_number
    :param x: N-D tensor with the data format
    :param divisible_number:
    :return: padding value at each dimension, e.g. 3D->(d3_p1,d3_p2, d2_p1, d2_p2, d1_p1, d1_p2)
    """
    padding = []
    input_shape_list = x.size()

    # Reversed because pytorch's pad() receive in a reversed order
    for org_size in reversed(input_shape_list[2:]):
        # compute the padding amount in two sides
        p = np.int32((np.int32((org_size - 1) / divisible_number) + 1) * divisible_number - org_size)
        # padding amount in one size
        p1 = np.int32(p / 2)
        padding.append(p1)
        padding.append(p - p1)

    return tuple(padding)


def norm_intensity(x, max_val=1):
    """
    This function normalizes the intensity of the input tensor x into [0, max_val]
    :param x: input tensor
    :param max_val: maximum value
    :return: a normalized tensor
    """
    if max_val is None:
        max_val = 1

    if len(list(torch.unique(input).size())) != 1:  # avoid the case that the tensor contains only one value
        x = x - torch.min(x)

    x = x / torch.max(x) * max_val
    return x


def interp(src_vol, loc_shifts, vol_size=None, interp_method="bilinear"):
    """
    This function implements the N-D gridded interpolation of a tensor src_vol
    with the shifted coordinate as loc_shift.
    :param vol_size: a tensor Size with a value of (1st_size, ..., nth_size)
    :param src_vol: a tensor with a size of (batch, 1, 1st_size, ..., nth_size)
    :param loc_shifts: a tensor with a size of (batch, n, 1st_size_out, ..., nth_size_out)
    :param interp_method: interpolation type "linear" (default) or "nearest"
    :return: interpolated volume of the same size as the size of loc_shifts
             (batch, 1, 1st_size_out, ..., nth_size_out)
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    if vol_size is None:
        vol_size = src_vol.shape[2:]

    # normalize grid values to [-1, 1] for resampler
    for i in range(len(vol_size)):
        loc_shifts[:, i, ...] = 2 * (loc_shifts[:, i, ...] / (vol_size[i] - 1) - 0.5)

    # permute the loc_shift into the form (batch, 1st_size, ..., nth_size, n)
    # to adapt the nn.functional.grid_sample() function
    if len(vol_size) == 2:
        loc_shift = loc_shifts.permute(0, 2, 3, 1)  # size of (batch, Hout, Wout, 2)
        loc_shift = loc_shift[..., [1, 0]]  # permute the two loc_shifts
    elif len(vol_size) == 3:
        loc_shifts = loc_shifts.permute(0, 2, 3, 4, 1)  # size of (batch, Dout, Hout, Wout, 3)
        loc_shifts = loc_shifts[..., [2, 1, 0]]  # permute the three loc_shifts
    
    return nn.functional.grid_sample(src_vol, loc_shifts, mode=interp_method)


# ------------------------------------------------------------------------------
# Classes for testing
# ------------------------------------------------------------------------------
class Test_conv_norm_layer(nn.Module):
    def __init__(self):
        super(Test_conv_norm_layer, self).__init__()
        self.norm_input = NormLayer(mode='pad', max_norm_val=1)
        self.conv = conv_block(1, 3, 3, kernel_size=7)
        self.recovery_size = NormLayer(mode='crop')
        # self.conv = nn.Conv3d(1, 2, kernel_size=5)

    def forward(self, x):
        print('x size input:', x.size())
        print('max(x) =', torch.max(x))
        x, padding = self.norm_input(x)
        print('x size after norm_input:', x.size())
        print('padding =', padding)
        print('max(x) =', torch.max(x))

        x = self.conv(x)
        print('x size after conv:', x.size())
        x = self.recovery_size(x, padding)
        print('x size after recovery_size:', x.size())
        return x


if __name__ == "__main__":
    # # For testing the UNet
    # x = torch.rand((1, 2, 90, 104, 72))
    # model = UNet(n_dims=3, n_dsp_fields=3, nf_enc=[16, 32, 32, 32], nf_dec=[32, 32, 32, 32, 8, 8],
    #              do_batchnorm=False, max_norm_val=None)
    # print(model)

    x = torch.rand((1, 1, 90, 104, 72)).cuda()
    # model = SNet(n_dims=3, n_dsp_fields=3, nf_enc=[16], nf_dec=[32, 8, 8],
    #              do_batchnorm=False, max_norm_val=None)
    model = SNet(n_dims=3, n_dsp_fields=3, nf_enc=[16, 32, 32, 32], nf_dec=[32, 32, 32, 32, 8, 8],
                 do_batchnorm=False, max_norm_val=None).cuda()
    print(model(x, x))

    # # For testing
    # input = torch.ones((1, 1, 189, 139, 60))
    # model = Test_conv_norm_layer()
    # a = model(input)
    #
    # print('max(x)=', torch.max(input))
    # b = norm_intensity(input, max_val=1)
    # print('max(b)=', torch.max(input))

    # print(a)
    # print(a.size())
    # b = nn.functional.pad(a, [1, 2])
    # c = nn.functional.pad(b, [-1, -2])
    # print(b.size())
    # print(c.size())
    # print(torch.all(torch.eq(a, c)))
    # b = nn.functional.pad(a, [1, 2, 3, 4])
    # c = nn.functional.pad(b, [-1, -2, -3, -4])
    # print(b.size())
    # print(c.size())
    # print(torch.all(torch.eq(a, c)))
    # b = nn.functional.pad(a, [1, 2, 3, 4, 5, 6])
    # c = nn.functional.pad(b, [-1, -2, -3, -4, -5, -6])
    # print(b.size())
    # print(c.size())
    # print(torch.all(torch.eq(a, c)))

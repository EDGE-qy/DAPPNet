#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import DCN

class NormPreserveDeformConvFunction(Function):
    def __init__(self):
        super(NormPreserveDeformConvFunction, self).__init__()
        self.stride = None
        self.padding = None
        self.dilation = None
    
    @staticmethod
    @torch.jit.script
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step, zero_padding=True):
        #assert input.shape[-3] % 2 == 0, "STANDING BUG: input must have an even number of channels"

        # ctx.stride = _pair(stride)
        # ctx.padding = _pair(padding)
        # ctx.dilation = _pair(dilation)
        if isinstance(stride, torch.Tensor):
            stride: List[int] = stride.tolist()  # Add type hint here
        if isinstance(padding, torch.Tensor):
            padding: List[int] = padding.tolist()  # Add type hint here
        if isinstance(dilation, torch.Tensor):
            dilation: List[int] = dilation.tolist()  # Add type hint here
        # if isinstance(stride, list):
        #     stride = torch.tensor(stride, dtype=torch.int)
        # if isinstance(padding, list):
        #     padding = torch.tensor(padding, dtype=torch.int)
        # if isinstance(dilation, list):
        #     dilation = torch.tensor(dilation, dtype=torch.int)
        # ctx.stride = torch.tensor(stride, dtype=torch.int)
        # ctx.padding = torch.tensor(padding, dtype=torch.int)
        # ctx.dilation = torch.tensor(dilation, dtype=torch.int)
        # ctx.kernel_size = _pair(weight.shape[2:4])
        # ctx.group = group
        # ctx.deformable_groups = deformable_groups
        # ctx.im2col_step = im2col_step
        # ctx.zero_padding = zero_padding
        stride_paired = _pair(stride)
        padding_paired = _pair(padding)
        dilation_paired = _pair(dilation)
        kernel_size_paired = _pair(weight.shape[2:4])
        output, deformed_columns = DCN.norm_preserve_deform_conv_forward(input, weight, bias,
                                         offset,
                                         #  ctx.kernel_size[0], ctx.kernel_size[1],
                                         #  ctx.stride[0], ctx.stride[1],
                                         #  ctx.padding[0], ctx.padding[1],
                                         #  ctx.dilation[0], ctx.dilation[1],
                                         kernel_size_paired[0], kernel_size_paired[1],
                                         stride_paired[0], stride_paired[1],
                                         padding_paired[0], padding_paired[1],
                                         dilation_paired[0], dilation_paired[1],
                                         group,
                                         deformable_groups,
                                         im2col_step,
                                         zero_padding)
        ctx.save_for_backward(input, offset, weight, bias, deformed_columns, stride, padding, dilation,
                              group, deformable_groups, im2col_step, zero_padding)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias, deformed_columns, stride, padding, dilation, group, deformable_groups, im2col_step, zero_padding = ctx.saved_tensors
        stride_paired = _pair(stride)
        padding_paired = _pair(padding)
        dilation_paired = _pair(dilation)
        kernel_size_paired = _pair(weight.shape[2:4])
        
        grad_input, grad_offset, grad_weight, grad_bias = \
            DCN.norm_preserve_deform_conv_backward(input, deformed_columns, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     #  ctx.kernel_size[0], ctx.kernel_size[1],
                                     #  ctx.stride[0], ctx.stride[1],
                                     #  ctx.padding[0], ctx.padding[1],
                                     #  ctx.dilation[0], ctx.dilation[1],
                                     kernel_size_paired[0], kernel_size_paired[1],
                                     stride_paired[0], stride_paired[1],
                                     padding_paired[0], padding_paired[1],
                                     dilation_paired[0], dilation_paired[1],
                                     group,
                                     deformable_groups,
                                     im2col_step,
                                     zero_padding)
        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None, None

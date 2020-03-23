from __future__ import division
from collections import namedtuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function


'''
block floating point :


'''

class BlockFP(InplaceFunction):
    @staticmethod
    def forward(ctx, input, bits, add_noise=True, block=False, block_dim=2):

        ctx.inplace = False
        if ctx.inplace:
            ctx.mark_dirty(input)
            x = input
        else:
            x = input.clone()

        with torch.no_grad():
            maxpos = pow(2, bits - 1) - 1
            maxneg = - pow(2, bits - 1)

            if x.abs().max() != 0:

                if block == False :
                    exp = torch.log2(x.abs().max()).ceil().sub(bits - 1)
                    scale = torch.pow(2, exp)

                else :
                    if block_dim == 2:
                        # {Out_Channel, Input_Channel for weight};
                        # {Batch, Input_Channel for Activation};
                        max_entry = torch.max(torch.abs(x.view(x.size(0) * x.size(1), -1)), 1)[0]
                        max_exponent = max_entry.add(1e-32).log2().ceil().view(
                            [x.size(0), x.size(1), 1, 1])
                        scale= torch.pow(2, (max_exponent.sub(bits-1)))
                    elif block_dim == 1 :
                        # {Out_Channel, for weight};
                        # {Batch,  for Activation};
                        max_entry = torch.max(torch.abs(x.view(x.size(0), -1)), 1)[0]
                        max_exponent = max_entry.add(1e-32).log2().ceil().view(
                            [x.size(0), 1, 1, 1])
                        scale= torch.pow(2, (max_exponent.sub(bits-1)))
                    else:
                        raise ValueError("invalid block dim option {}".format(block_dim))

                if add_noise:
                    noise = torch.rand_like(x)
                    x.div_(scale).add_(noise).floor_().clamp_(maxneg, maxpos).mul_(scale)
                else:
                    x.div_(scale).round_().clamp_(maxneg, maxpos).mul_(scale)

        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = grad_outputs
        return grad_inputs, None, None, None, None

class BlockFP_Grad(InplaceFunction):
    @staticmethod
    def forward(ctx, input, bits=16, add_noise=False, block=False, block_dim=2):
        ctx.bits        = bits
        ctx.add_noise   = add_noise
        ctx.block       = block
        ctx.block_dim   = block_dim

        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        bits       =  ctx.bits
        add_noise  =  ctx.add_noise
        block      =  ctx.block
        block_dim  =  ctx.block_dim

        grad_inputs = num2bfp(grad_outputs, bits, add_noise, block, block_dim)

        return  grad_inputs, None, None, None, None



def num2bfp(x, bits, add_noise=False, block=False, block_dim=2):
    qx = BlockFP().apply(x, bits, add_noise, block, block_dim)
    return qx

def bfp_grad(x, bits, add_noise=False, block=False, block_dim=2):
    qx = BlockFP_Grad().apply(x, bits, add_noise, block, block_dim)
    return qx


def bfp_fb(x, bits_f = 16, bits_b=16, add_noise=False, block=False, block_dim=2):
    x = num2bfp(x, bits_f, add_noise, block, block_dim)
    x = bfp_grad(x, bits_b, False, block, block_dim)
    return  x






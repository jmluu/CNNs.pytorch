import torch
from torch.autograd.function import InplaceFunction

class LowFloat(InplaceFunction):
    """
    float8 (1, 5, 2)
    0 ： e = '00000' and f = 0
    Inf: e='11111', f=0
    NaN: e='11111' f!=0
    normal : 2^{e-11}*(1.f)
        max-normal = 2**(30-15)*(1.75)
        min-normal = 2**(1-15)
    subnormal:    e='00000' 2**(-14)*(0.f)
    """
    @staticmethod
    def forward(ctx, input, num_bits=8, exp_bits=5, mant_bits=2, inplace=False):
        assert  num_bits == (exp_bits + mant_bits + 1)
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            x = input
        else:
            x = input.clone()

        with torch.no_grad():
            eps = float(1e-32)
            mid_exp = 2**(exp_bits-1)-1
            max_exp = 2**(exp_bits)-2 - mid_exp
            min_exp = 1 - mid_exp

            max_mant = (2**(mant_bits)-1)/(2**(mant_bits))
            int_mant = 2**mant_bits
            maxnorm = float(2**(max_exp)*(1+max_mant))
            minnorm = float(2**(min_exp))
            minsub  = float(2**(min_exp)*(2**(-mant_bits)))

            x = torch.where(x.abs().ge(minsub), x, torch.zeros_like(x))
            # extract information
            sign = x.sign()
            nzero = torch.ne(x, 0.0).to(torch.float32)
            x.abs_()
            x.clamp_(minsub, maxnorm)

            is_norm = torch.ge(x, minnorm)

            e = torch.log2(x+eps).floor()
            f = x.div(torch.pow(2, e)).sub(1.0)  # fraction part value, float, [1, 2)
            # norm case
            norm_x = torch.pow(2, e).mul(
                f.mul(int_mant).round().div(int_mant).add(1.0))

            # sub-normal case
            sub_x = x.div(2**min_exp).mul(int_mant).round().div(int_mant).mul(2**min_exp)
            output = torch.where(is_norm, norm_x, sub_x)*sign*nzero
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None

def num2lowfloat(x, num_bits=16, exp_bits=5, mant_bits=10, inplace=False ):
    return LowFloat().apply(x, num_bits, exp_bits, mant_bits, inplace)


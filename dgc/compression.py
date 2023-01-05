import copy
import math
import random

import torch

import horovod.torch as hvd
from horovod.torch.mpi_ops import Average
from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async as allgather_async_
from horovod.torch.mpi_ops import synchronize as synchronize_

from dgc.memory import Memory

__all__ = ['DGCCompressor']


class DGCCompressor:
    def __init__(self, compress_ratio, memory=None,
                 sample_ratio=0.01, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=0.8, max_adaptation_iters=10, resample=True,
                 fp16_values=False, int32_indices=False,
                 warmup_epochs=-1, warmup_coeff=None, snr_compression=False, snr_warmup=False,
                 beta1=0.9, beta2=0.995, beta1_warmup=False, l_t=None, bin_multiplier=2, snr_init='zeros',
                sq_init_factor=1.0, init_snr_after_warmup=False, use_bias_correction=False):
        self.world_size = hvd.size()
        self.op = Average
        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1

        self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.resample = resample

        self.attributes = {}
        self.state = {}
        self.epoch = 0
        # Use SNR compression if True else use DGC
        self.snr_compression = snr_compression 
        # Use SNR compression during warmup phase, else use DGC
        self.snr_warmup = snr_warmup
        # Beta values for computing moving SNR
        self.beta1 = beta1
        self.beta2 = beta2
        # Ramp up the beta1 value if True
        self.beta1_warmup = beta1_warmup
        # bin size for filtering snr using neighboring values
        self.l_t = l_t
        self.bin_multiplier = bin_multiplier
        self.snr_init = snr_init # init in {'zeros|ones|1e-8|grad_init'}
        self.sq_init_factor = sq_init_factor # multiplier for the snr second moment initializer
        self.init_snr_after_warmup = init_snr_after_warmup
        self.use_bias_correction = use_bias_correction

    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            if self.sample_ratio < 1.0:
                pct_numel = int(math.ceil(numel * self.sample_ratio))
                cpr_numel = int(math.ceil(2 / self.compress_ratio))
                if numel <= cpr_numel:
                    if hvd.rank() == 0:
                        print(f'Warning: {name} with {numel} elements transmits 1 gradient element')
                    sample_stride = 1
                    num_samples = numel
                else:
                    sample_stride = int(math.ceil(numel / max(pct_numel, cpr_numel) / 32)) * 32 + 1
                    num_samples = numel // sample_stride
                    while num_samples < max(pct_numel, cpr_numel):
                        sample_stride = sample_stride - 8
                        num_samples = numel // sample_stride
            else:
                sample_stride = 1
                num_samples = numel
            top_k_samples = int(math.ceil(num_samples * self.compress_ratio))
            num_selects = int(math.ceil(numel * self.compress_ratio))
            self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples, sample_stride)
            if hvd.rank() == 0:
                print(f'   {name:<25}: transmit {num_selects} / {numel} elements of shape {shape}\n'
                      f'   {" " * 25}  threshold {top_k_samples} / {num_samples} samples'
                      f' {f"at stride {sample_stride}" if self.strided_sample else "uniformly"}')
    
    def warmup_compress_ratio(self, epoch):
        self.epoch = epoch
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                        self.base_compress_ratio)
            else:
                compress_ratio = self.base_compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            if hvd.rank() == 0:
                print(f'update compress ratio: {compress_ratio}')
            self.compress_ratio = compress_ratio
            self.initialize(self.attributes.items())

    def initialize_snr(self, grad):
        # zeros|ones|1e-8|grad_init
        if self.snr_init == 'zeros':
            return torch.zeros_like(grad), torch.zeros_like(grad)
        elif self.snr_init == 'ones':
            return torch.ones_like(grad), torch.ones_like(grad)
        elif self.snr_init == '1e-8':
            return 1e-8*torch.ones_like(grad), 1e-8*torch.ones_like(grad)
        elif self.snr_init == 'grad_init':
            return copy.deepcopy(grad), copy.deepcopy(grad) * copy.deepcopy(grad)
        elif self.snr_init == 'av_grad_init':
            return copy.deepcopy(grad), 1/(torch.abs(copy.deepcopy(grad)) + 1e-8) # torch.rand_like(grad) + 1e-4 #self.sq_init_factor*torch.ones_like(grad)
        else:
            raise ValueError("snr_init must be in {`zeros`|`ones`|`1e-8`|`grad_init`|`av_grad_init`}")

    def _sparsify(self, tensor, name, step=0):
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples, sample_stride = self.attributes[name]

        importance = tensor.abs()
        if numel == num_samples:
            samples = importance
        else:
            if self.strided_sample:
                sample_start = random.randint(0, sample_stride - 1)
                samples = importance[sample_start::sample_stride]
            else:
                samples = importance[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        # SNR
        do_snr_compression = self.snr_compression
        beta1 = self.beta1
        beta2 = self.beta2
        if self.epoch < self.warmup_epochs:
            do_snr_compression = self.snr_warmup
            # warmup beta1
            if self.beta1_warmup:
                beta1 = max(0.001, self.beta1*self.epoch/self.warmup_epochs)

        qs = [0, .1, .3, .5, .8, .9, .999, .9995, .9999, .99999]
        grad = tensor.data
#         init_snr = (not self.init_snr_after_warmup) or ( )
#         if hvd.rank() == 0:
#             if name == 'conv1.weight':
#                 print('name in state:', name in self.state)
        if name not in self.state:
            avg, sq = self.initialize_snr(grad)
#             avg, sq = copy.deepcopy(grad), 1/(torch.abs(copy.deepcopy(grad)) + 1e-8) #torch.rand_like(grad) + 1e-4
#             avg, sq = grad, torch.ones_like(grad)
            debug = {"bin_compress_ratio": 1, "bin_disparity": -1, "bin_max": -1, "bin_median": -1}
            self.state[name] = {"exp_avg": avg, "exp_avg_sq": sq, "debug": debug, "reinitialized": False}
        
        state = self.state[name]
        
        if self.init_snr_after_warmup and (self.epoch >= self.warmup_epochs):
            step -= 100*self.warmup_epochs
            if not state["reinitialized"]:
                avg, sq = self.initialize_snr(grad) # re-initialize
                state["exp_avg"] = avg
                state["exp_avg_sq"] = sq
                state["reinitialized"] = True # only reinitialize once
            
        
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        snr = torch.abs(torch.div(exp_avg, torch.sqrt(exp_avg_sq + 1e-8)))
        state['debug']['snr_median'] = torch.median(snr)
        state['debug']['snr_max'] = torch.max(snr)
        quants = torch.quantile(snr, torch.tensor(qs).to(tensor.device))
        for q in range(len(qs)):
            state['debug'][f'snr_quantile/snr_{qs[q]}'] = quants[q]
        quants = torch.quantile(exp_avg, torch.tensor(qs).to(tensor.device))
        for q in range(len(qs)):
            state['debug'][f'exp_avg_quantile/exp_avg_{qs[q]}'] = quants[q]
        quants = torch.quantile(exp_avg_sq, torch.tensor(qs).to(tensor.device))
        for q in range(len(qs)):
            state['debug'][f'exp_avg_sq_quantile/exp_avg_sq_{qs[q]}'] = quants[q]
        state['debug']['compress_ratio'] = self.compress_ratio
        
        # include stats about the gradient
        quants = torch.quantile(torch.abs(grad), torch.tensor(qs).to(tensor.device))
        for q in range(len(qs)):
            state['debug'][f'grad_quantile/grad_{qs[q]}'] = quants[q]
        if do_snr_compression:
            importance = snr
            samples = snr
        #top_k_samples = (0.1*top_k_samples).int()

        ## END SNR

        if self.l_t is not None:
#             if self.epoch < 3: # state["first_step"] == 1:
#                 mask = torch.ones_like(grad).long()
                # state["first_step"] = 0
#                 threshold = 0
#                 state['bin_compress_ratio'] = mask.float().mean() # compression ratio of alpha means we only send alpha percent of weights
                # TODO: remove
#                 mask_shape = importance.shape
#                 importance = importance.reshape((-1, self.l_t))
#                 bin_max, _ = torch.max(importance, axis=1)
#                 bin_max = bin_max.reshape((-1, 1)) # reshape to compatible dimensionality for binned to_mask matrix for comparison
#                 mask = torch.ge(self.bin_multiplier * importance, bin_max).reshape(mask_shape)
#                 state['debug']['bin_compress_ratio'] = mask.float().mean()
# #                 state['debug']['bin_disparity'] = torch.median(torch.abs(bin_max - importance.median(axis=1)[0]))
#                 state['debug']['bin_max'] = torch.median(bin_max)
#                 state['debug']['bin_median'] = torch.median(torch.median(importance, axis=1)[0])
#                 state['debug']['bin_disparity'] = state['debug']['bin_max'] - state['debug']['bin_median']
            
#                 mask = torch.ones_like(grad).long()
#                 # state["first_step"] = 0
#                 threshold = 0
#             else:
            mask_shape = importance.shape
            importance = importance.reshape((-1, self.l_t))
            bin_max, _ = torch.max(importance, axis=1)
            bin_max = bin_max.reshape((-1, 1)) # reshape to compatible dimensionality for binned to_mask matrix for comparison
            if self.epoch < 7: # state["first_step"] == 1:
                mask = torch.ones_like(grad).long()
            else:
                mask = torch.gt(self.bin_multiplier * importance, bin_max).reshape(mask_shape)
            state['debug']['bin_compress_ratio'] = mask.float().mean()
#                 state['debug']['bin_disparity'] = torch.median(torch.abs(bin_max - importance.median(axis=1)[0]))
            state['debug']['bin_max'] = torch.median(bin_max)
            state['debug']['bin_median'] = torch.median(torch.median(importance, axis=1)[0])
            state['debug']['bin_disparity'] = state['debug']['bin_max'] - state['debug']['bin_median']
#                 state['debug']['bin_median'] = importance.median() #torch.median(torch.median(importance, axis=1)[0])[0]
#                 print(state['debug']['bin_disparity'])
            # print('importance cont', importance.max(), bin_max.max())
            threshold = 0 if importance.sum() == 0 else torch.min(importance.reshape(-1)[mask.reshape(-1)])
            print('threshold', threshold, 'snr', snr.sum(), 'bin_max', bin_max.max(), 'importance_median', importance.median())
        else:
            threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
            mask = torch.ge(importance, threshold)

        # Decay the first and second moment running average coefficient
#         if do_snr_compression:
#         state['debug']['snr_top_min'] = threshold
        exp_avg[mask] = self.beta1 * exp_avg[mask] + (1 - self.beta1) * grad[mask]
        exp_avg_sq[mask] = self.beta2 * exp_avg_sq[mask] + (1 - self.beta2) * grad[mask]*grad[mask] 

        if self.use_bias_correction:
            exp_avg[mask] = exp_avg[mask]/(1 - self.beta1**step)
            exp_avg_sq[mask] = exp_avg_sq[mask]/(1 - self.beta2**step)
            
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()

        #if numel > num_samples:
        #    # code modified from https://github.com/sands-lab/grace/blob/master/grace_dl/torch/compressor/dgc.py
        #    for _ in range(self.max_adaptation_iters):
        #        if num_indices > num_selects:
        #            if num_indices > num_selects * self.compress_upper_bound:
        #                if self.resample:
        #                    indices = indices[
        #                        torch.topk(importance[indices], num_selects,
        #                                   0, largest=True, sorted=False)[1]
        #                    ]
        #                    break
        #                else:
        #                    threshold = threshold * self.compress_upper_bound
        #            else:
        #                break
        #        elif num_indices < self.compress_lower_bound * num_selects:
        #            threshold = threshold * self.compress_lower_bound
        #        else:
        #            break
        #        mask = torch.ge(importance, threshold)
        #        indices = mask.nonzero().view(-1)
        #        num_indices = indices.numel()

        #indices = indices[:num_selects]
        indices = indices[:mask.sum().int()] # TODO
        values = tensor[indices]
        
        if hvd.rank() == 0:
            if name == 'conv1.weight':
                # print(name, torch.median(torch.abs(grad)))
                print('================================================')
                print(' one count', mask.sum(), 'threshold', threshold, 'top_k_samples', top_k_samples)
                print(' indices', indices, 'values', values)
                print(' snr', torch.abs(torch.div(exp_avg, torch.sqrt(exp_avg_sq + 1e-8)))[indices])
                print(' exp_avg', exp_avg[indices], 'exp_avg_sq', exp_avg_sq[indices])
        indices = indices[:num_selects] # TODO
        values = tensor[indices]
        return values, indices, numel, shape, num_selects

    def compress(self, tensor, name):
        if self.compress_ratio < 2.0 and name in self.attributes:
            # compress
            tensor_compensated = self.memory.compensate(
                tensor, name, accumulate=True)
            values, indices, numel, shape, num_selects = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices, ))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))
            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            return (values, indices), ctx
        else:
            ctx = (name, None, None, tensor.dtype, None, None)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)
            if self.op == Average:
                grad.mul_(1. / self.world_size)
            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)

    def communicate(self, tensor_compressed, name, op):
        self.op = op
        if self.compress_ratio < 1.0 and name in self.attributes:
            return [allgather_async_(t, name=f'{name}.t{e}')
                    for e, t in enumerate(tensor_compressed)]
        else:
            return allreduce_async_(tensor_compressed, name=name, op=op)

    def synchronize(self, handle):
        if isinstance(handle, (tuple, list)):
            return [synchronize_(h) for h in handle]
        else:
            return synchronize_(handle)

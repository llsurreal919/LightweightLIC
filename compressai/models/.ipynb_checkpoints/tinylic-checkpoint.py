import time
import math
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import RSTB, MultistageMaskedConv2d
from timm.models.layers import trunc_normal_
from compressai.models.elan_block import ELAB
from .utils import conv, deconv, update_registered_buffers, Demultiplexer, Multiplexer

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TinyLIC_LK(nn.Module):
    """

    """

    def __init__(self, N=129, M=192):
        super().__init__()

        self.window_sizes = [4, 8, 16]
        self.c_elan  = 129
        self.n_share = 0
        self.r_expand = 2
        self.num_blk = [2,4,6,2,2,2]

        # define head module
        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        m_g_a1 = []
        for i in range(self.num_blk[0]):#24 1
            if (i+1) % 2 == 1: 
                m_g_a1.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_a1.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        m_g_a3 = []
        for i in range(self.num_blk[1]):#24 1
            if (i+1) % 2 == 1: 
                m_g_a3.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_a3.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        m_g_a5 = []
        for i in range(self.num_blk[2]):#24 1
            if (i+1) % 2 == 1: 
                m_g_a5.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_a5.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        m_g_a7 = []
        for i in range(self.num_blk[3]):#24 1
            if (i+1) % 2 == 1: 
                m_g_a7.append(
                    ELAB(
                        M, M, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_a7.append(
                    ELAB(
                        M, M, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        m_h_a1 = []
        for i in range(self.num_blk[4]):#24 1
            if (i+1) % 2 == 1: 
                m_h_a1.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        [4,8,8], shared_depth=self.n_share
                    )
                )
            else:              
                m_h_a1.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        [4,8,8], shared_depth=self.n_share
                    )
                )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        m_h_a3 = []
        for i in range(self.num_blk[5]):#24 1
            if (i+1) % 2 == 1: 
                m_h_a3.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        [2,4,4], shared_depth=self.n_share
                    )
                )
            else:              
                m_h_a3.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        [2,4,4], shared_depth=self.n_share
                    )
                )

        self.num_blk = self.num_blk[::-1]

        m_h_s0 = []
        for i in range(self.num_blk[0]):#24 1
            if (i+1) % 2 == 1: 
                m_h_s0.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        [2,4,4], shared_depth=self.n_share
                    )
                )
            else:              
                m_h_s0.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        [2,4,4], shared_depth=self.n_share
                    )
                )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        m_h_s2 = []
        for i in range(self.num_blk[1]):#24 1
            if (i+1) % 2 == 1: 
                m_h_s2.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        [4,8,8], shared_depth=self.n_share
                    )
                )
            else:              
                m_h_s2.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        [4,8,8], shared_depth=self.n_share
                    )
                )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)
        
        m_g_s0 = []
        for i in range(self.num_blk[2]):#24 1
            if (i+1) % 2 == 1: 
                m_g_s0.append(
                    ELAB(
                        M, M, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_s0.append(
                    ELAB(
                        M, M, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        m_g_s2 = []
        for i in range(self.num_blk[3]):#24 1
            if (i+1) % 2 == 1: 
                m_g_s2.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_s2.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        m_g_s4 = []
        for i in range(self.num_blk[4]):#24 1
            if (i+1) % 2 == 1: 
                m_g_s4.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_s4.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        m_g_s6 = []
        for i in range(self.num_blk[5]):#24 1
            if (i+1) % 2 == 1: 
                m_g_s6.append(
                    ELAB(
                        N, N, self.r_expand, 0, #shift = 0
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_g_s6.append(
                    ELAB(
                        N, N, self.r_expand, 1, #shift = 1
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        self.context_prediction_1 = MultistageMaskedConv2d(
                M, M*2, kernel_size=3, padding=1, stride=1, mask_type='A'
        )
        self.context_prediction_2 = MultistageMaskedConv2d(
                M, M*2, kernel_size=3, padding=1, stride=1, mask_type='B'
        )
        self.context_prediction_3 = MultistageMaskedConv2d(
                M, M*2, kernel_size=3, padding=1, stride=1, mask_type='C'
        )

        self.entropy_parameters = nn.Sequential(
                conv(M*24//3, M*18//3, 1, 1),
                nn.GELU(),
                conv(M*18//3, M*12//3, 1, 1),
                nn.GELU(),
                conv(M*12//3, M*6//3, 1, 1),
        ) 

        self.apply(self._init_weights)
        self.g_a1 = nn.Sequential(*m_g_a1)  
        self.g_a3 = nn.Sequential(*m_g_a3)
        self.g_a5 = nn.Sequential(*m_g_a5)
        self.g_a7 = nn.Sequential(*m_g_a7)

        self.g_s0 = nn.Sequential(*m_g_s0)  
        self.g_s2 = nn.Sequential(*m_g_s2)
        self.g_s4 = nn.Sequential(*m_g_s4)
        self.g_s6 = nn.Sequential(*m_g_s6)

        self.h_s0 = nn.Sequential(*m_h_s0)  
        self.h_s2 = nn.Sequential(*m_h_s2)

        self.h_a1 = nn.Sequential(*m_h_a1)  
        self.h_a3 = nn.Sequential(*m_h_a3)

        self.Gain_context = torch.nn.Parameter(torch.ones(size=[1, M*2]), requires_grad=True)
        self.Gain = torch.nn.Parameter(torch.ones(size=[1, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[1, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[1, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[1, N]), requires_grad=True)

    def g_a(self, x, x_size=None):
        x = self.g_a0(x)
        x = self.g_a1(x) + x
        x = self.g_a2(x)
        x = self.g_a3(x) + x
        x = self.g_a4(x)
        x = self.g_a5(x) + x
        x = self.g_a6(x)
        x = self.g_a7(x) + x
        return x

    def g_s(self, x, x_size=None):
        x = self.g_s0(x) + x
        x = self.g_s1(x)
        x = self.g_s2(x) + x
        x = self.g_s3(x)
        x = self.g_s4(x) + x
        x = self.g_s5(x)
        x = self.g_s6(x) + x
        x = self.g_s7(x)
        return x

    def h_a(self, x, x_size=None):
        x = self.h_a0(x)
        x = self.h_a1(x) + x
        x = self.h_a2(x)
        x = self.h_a3(x) + x
        return x

    def h_s(self, x, x_size=None):
        x = self.h_s0(x) + x
        x = self.h_s1(x)
        x = self.h_s2(x) + x
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        start = time.time()
        y = self.g_a(x)
        y = y * torch.abs(self.Gain).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1] 192
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain).unsqueeze(2).unsqueeze(3) 
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain).unsqueeze(2).unsqueeze(3)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        y_1 = y_hat.clone()
        y_1[:, :, 0::2, 1::2] = 0
        y_1[:, :, 1::2, :] = 0
        ctx_params_1 = self.context_prediction_1(y_1)
        ctx_params_1 = ctx_params_1 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3) ##对每一层的上下文进行通道自适应？
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0

        y_2 = y_hat.clone()
        y_2[:, :, 0::2, 1::2] = 0
        y_2[:, :, 1::2, 0::2] = 0
        ctx_params_2 = self.context_prediction_2(y_2)
        ctx_params_2 = ctx_params_2 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3)
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0

        y_3 = y_hat.clone()
        y_3[:, :, 1::2, 0::2] = 0
        ctx_params_3 = self.context_prediction_3(y_3)
        ctx_params_3 = ctx_params_3 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3)
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        end = time.time()
        runTime = end - start
        # print("压缩时间：", runTime, "秒")
        start = time.time()
        y_hat = y_hat * torch.abs(self.InverseGain).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)
        end = time.time()
        runTime1 = end - start
        # print("解压时间：", runTime, "秒")

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "enc-time":runTime,
            "dec-time":runTime1
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y = y * torch.abs(self.Gain).unsqueeze(2).unsqueeze(3)
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain).unsqueeze(2).unsqueeze(3) # 通道自适应
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * torch.abs(self.InverseHyperGain).unsqueeze(2).unsqueeze(3)# 通道自适应

        params = self.h_s(z_hat)

        zero_ctx_params = torch.zeros_like(params).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params, zero_ctx_params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)
        
        y_1 = y_hat.clone()
        y_1[:, :, 0::2, 1::2] = 0
        y_1[:, :, 1::2, :] = 0
        ctx_params_1 = self.context_prediction_1(y_1)
        ctx_params_1 = ctx_params_1 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3) ##对每一层的上下文进行通道自适应？
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0
        
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, zero_ctx_params, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        y_2 = y_hat.clone()
        y_2[:, :, 0::2, 1::2] = 0
        y_2[:, :, 1::2, 0::2] = 0
        ctx_params_2 = self.context_prediction_2(y_2)
        ctx_params_2 = ctx_params_2 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3)# 通道自适应
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, zero_ctx_params), dim=1)
        )
        _, means_hat = gaussian_params.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat)

        y_3 = y_hat.clone()
        y_3[:, :, 1::2, 0::2] = 0
        ctx_params_3 = self.context_prediction_3(y_3)
        ctx_params_3 = ctx_params_3 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3)# 通道自适应
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0

        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        
        y1, y2, y3, y4 = Demultiplexer(y)
        scales_hat_y1, scales_hat_y2, scales_hat_y3, scales_hat_y4 = Demultiplexer(scales_hat)
        means_hat_y1, means_hat_y2, means_hat_y3, means_hat_y4 = Demultiplexer(means_hat)

        start = time.time()
        indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat_y1)
        indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat_y2)
        indexes_y3 = self.gaussian_conditional.build_indexes(scales_hat_y3)
        indexes_y4 = self.gaussian_conditional.build_indexes(scales_hat_y4)

        y1_strings = self.gaussian_conditional.compress(y1, indexes_y1, means=means_hat_y1)
        y2_strings = self.gaussian_conditional.compress(y2, indexes_y2, means=means_hat_y2)
        y3_strings = self.gaussian_conditional.compress(y3, indexes_y3, means=means_hat_y3)
        y4_strings = self.gaussian_conditional.compress(y4, indexes_y4, means=means_hat_y4)
        end = time.time()
        runTime = end - start
        # print("压缩减去时间：", runTime, "秒")
        return {
            "strings": [y1_strings, y2_strings, y3_strings, y4_strings, z_strings],
            "shape": z.size()[-2:],
            "compress-time": runTime
        }
    
    def decompress(self, strings, shape):
        """
        See Figure 5. Illustration of the proposed two-pass decoding.
        """
        assert isinstance(strings, list) and len(strings) == 5
        start = time.time()
        z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        end = time.time()
        runTime = end - start
        z_hat = z_hat * torch.abs(self.InverseHyperGain).unsqueeze(2).unsqueeze(3)# 通道自适应
        params = self.h_s(z_hat)

        # Stage 0:
        zero_ctx_params = torch.zeros_like(params).to(z_hat.device)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, zero_ctx_params, zero_ctx_params, zero_ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat_y1, _, _, _ = Demultiplexer(scales_hat)
        means_hat_y1, _, _, _ = Demultiplexer(means_hat)

        start1 = time.time()
        indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat_y1)
        _y1 = self.gaussian_conditional.decompress(strings[0], indexes_y1, means=means_hat_y1)     # [1, 384, 8, 8]
        end1 = time.time()
        runTime1 = end1 - start1
        y1 = Multiplexer(_y1, torch.zeros_like(_y1), torch.zeros_like(_y1), torch.zeros_like(_y1))    # [1, 192, 16, 16]
        
        # Stage 1:
        ctx_params_1 = self.context_prediction_1(y1)
        ctx_params_1 = ctx_params_1 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3) ##对每一层的上下文进行通道自适应？
        ctx_params_1[:, :, 0::2, :] = 0
        ctx_params_1[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, zero_ctx_params, zero_ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, scales_hat_y2, _, _ = Demultiplexer(scales_hat)
        _, means_hat_y2, _, _ = Demultiplexer(means_hat)

        start2 = time.time()
        indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat_y2)
        _y2 = self.gaussian_conditional.decompress(strings[1], indexes_y2, means=means_hat_y2)     # [1, 384, 8, 8]
        end2 = time.time()
        runTime2 = end2 - start2
        y2 = Multiplexer(torch.zeros_like(_y2), _y2, torch.zeros_like(_y2), torch.zeros_like(_y2))    # [1, 192, 16, 16]

        # Stage 2:
        ctx_params_2 = self.context_prediction_2(y1 + y2)
        ctx_params_2 = ctx_params_2 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3)# 通道自适应
        ctx_params_2[:, :, 0::2, 0::2] = 0
        ctx_params_2[:, :, 1::2, :] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, zero_ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, _, scales_hat_y3, _ = Demultiplexer(scales_hat)
        _, _, means_hat_y3, _ = Demultiplexer(means_hat)

        start3 = time.time()
        indexes_y3 = self.gaussian_conditional.build_indexes(scales_hat_y3)
        _y3 = self.gaussian_conditional.decompress(strings[2], indexes_y3, means=means_hat_y3)     # [1, 384, 8, 8]
        end3 = time.time()
        runTime3= end3 - start3
        y3 = Multiplexer(torch.zeros_like(_y3), torch.zeros_like(_y3), _y3, torch.zeros_like(_y3))    # [1, 192, 16, 16]

        # Stage 3:
        ctx_params_3 = self.context_prediction_3(y1 + y2 + y3)
        ctx_params_3 = ctx_params_3 * torch.abs(self.Gain_context).unsqueeze(2).unsqueeze(3)# 通道自适应
        ctx_params_3[:, :, 0::2, :] = 0
        ctx_params_3[:, :, 1::2, 1::2] = 0
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params_1, ctx_params_2, ctx_params_3), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, _, _, scales_hat_y4 = Demultiplexer(scales_hat)
        _, _, _, means_hat_y4 = Demultiplexer(means_hat)

        start4 = time.time()
        indexes_y4 = self.gaussian_conditional.build_indexes(scales_hat_y4)
        _y4 = self.gaussian_conditional.decompress(strings[3], indexes_y4, means=means_hat_y4)     # [1, 384, 8, 8]
        end4 = time.time()
        runTime4 = end4 - start4
        y4 = Multiplexer(torch.zeros_like(_y4), torch.zeros_like(_y4), torch.zeros_like(_y4), _y4)    # [1, 192, 16, 16]
        # print("解压减去时间：", runTime1+runTime2+runTime3+runTime4, "秒")
        # gather
        y_hat = y1 + y2 + y3 + y4
        y_hat = y_hat * torch.abs(self.InverseGain).unsqueeze(2).unsqueeze(3)# 通道自适应
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat,
            "decompress-time":runTime + runTime1 + runTime2 + runTime3 + runTime4
        }
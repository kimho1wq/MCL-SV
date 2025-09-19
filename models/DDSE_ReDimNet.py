
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import models.layers.poolings as pooling_layers
from models.layers.layernorm import LayerNorm
from models.layers.layernorm import LayerNorm
from models.layers.convnext import ConvNeXtLikeBlock
from models.layers.attention import TransformerEncoderLayer
from models.layers.redim_structural import to1d, to2d, weigth1d


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, reduction=8, n_group=4):
        super(SEBasicBlock, self).__init__()
        self.gnorm1 = nn.GroupNorm(n_group, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.gnorm2 = nn.GroupNorm(n_group, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

        self.se = SELayer(planes, reduction)

        if inplanes == planes:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)
    

    def forward(self, x):
        residual = x

        x = self.gnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.gnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        x = self.se(x)

        return x + self.residual_layer(residual)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class TimeContextBlock1d(nn.Module):
    def __init__(self, C, hC):
        super(TimeContextBlock1d, self).__init__()
        
        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C,hC,1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first")
        )
        self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
                TransformerEncoderLayer(
                    n_state=hC, 
                    n_mlp=hC, 
                    n_head=4
                )
            )
        self.exp_dim_conv = nn.Conv1d(hC,C,1)

    def forward(self,x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x
 
    
class Upsample(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.stride, mode='nearest')
    
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)

class STFTtoMelBanks(nn.Module):
    def __init__(self, 
        sample_rate = 16000, 
        n_fft = 512, 
        f_min = 20, 
        f_max = 7600, 
        n_mels = 72, 
    ):
        super(STFTtoMelBanks, self).__init__()
        
        self.spec_norm = lambda x : x - torch.mean(x, dim=-1, keepdim=True)
        mel_scale = torchaudio.transforms.MelScale(
            n_stft      = n_fft // 2,# + 1,
            n_mels      = n_mels,
            sample_rate = sample_rate,
            f_min       = f_min,
            f_max       = f_max,
        )
        self.register_buffer("mel_mat", mel_scale.fb.transpose(0, 1)) 

        
    def forward(self, x):
        mag = x.abs() ** 2        # (B, F, T)

        mel_spec = torch.matmul(self.mel_mat, mag)   # (B, M, T)

        log_mel = (mel_spec + 1e-6).log()
        log_mel = self.spec_norm(log_mel)

        return log_mel    

class STFT(nn.Module):
    def __init__(self, hop_length):
        super(STFT, self).__init__()
        self.stft = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.Spectrogram(
            n_fft        = 512,
            hop_length   = hop_length,
            win_length   = 400,
            window_fn    = torch.hamming_window,
            center       = True,
            pad_mode     = "reflect",
            power        = None,         
            )
        )

    def forward(self, x):
        return torch.view_as_real(self.stft(x)[:, :-1, :]).permute(0, 3, 1, 2) #  (B, 2, F, T) 


class DDSE_ReDimNet(nn.Module): 
    def __init__(self, stages_setup, C, F, embed_dim, group_divisor=4, hop_length=160,
        out_channels=None, pooling_func='ASTP', global_context_att=True, **kwargs):
        super(DDSE_ReDimNet, self).__init__()
  
        self.l_channel = [16, 16, 32, 32]
        self.l_num_convblocks = [3, 3, 4, 4]
        self.code_dim = embed_dim
        self.stride = [1,2,1,2]
        self.first_ks = 7
        self.first_st = (2,1)
        self.first_pa = 3

        self.stft = STFT(hop_length)
        self.stfttomel = STFTtoMelBanks(n_mels=F)
        
        self.n_level = len(self.l_channel)
        self.conv1 = nn.Conv2d(2, self.l_channel[0], kernel_size=self.first_ks, stride=self.first_st, padding=self.first_pa)
       

        ######### Description of Frontend #########
        for i in range(self.n_level):
            in_channel = self.l_channel[i] if i == 0 else self.l_channel[i-1]
            layers = [nn.Conv2d(in_channel, self.l_channel[i], kernel_size=3, stride=self.stride[i], padding=1)]
            layers.extend(
                [SEBasicBlock(self.l_channel[i], self.l_channel[i])
                 for _ in range(self.l_num_convblocks[i])]
            )
            setattr(self, f'd_res_{i+1}', nn.Sequential(*layers))


        for k in ['se', 'ne']:
            setattr(self, f'bottleneck_{k}', nn.Sequential(*[
                SEBasicBlock(self.l_channel[self.n_level-1], self.l_channel[self.n_level-1])
                for _ in range(4)
            ]))
        
            for i in range(self.n_level-1, -1, -1):
                layers = [SEBasicBlock(self.l_channel[i]*2, self.l_channel[i])]
                layers.extend(
                    [SEBasicBlock(self.l_channel[i], self.l_channel[i])
                    for _ in range(self.l_num_convblocks[i]-1)]
                )
                if self.stride[i] > 1:
                    layers.append(Upsample(self.stride[i]))
                
                out_channel = self.l_channel[i] if i == 0 else self.l_channel[i-1]
                layers.append(nn.Conv2d(self.l_channel[i], out_channel, kernel_size=3, padding=1))
                setattr(self, f'u_res_{k}_{i+1}', nn.Sequential(*layers))

            setattr(self, f'conv2_{k}', nn.ConvTranspose2d(self.l_channel[0]*2, 2, kernel_size=self.first_st, stride=self.first_st))

        

        ######### Description of Backend #########
        self.n_level_redimnet = len(stages_setup)

        self.stem3 = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, stride=1, padding='same'),
            LayerNorm(C, eps=1e-6, data_format="channels_first"),
            to1d()
        )

        c = C
        f = F
        for i, (stride, num_blocks, att_block_red) in enumerate(stages_setup):
            num_feats_to_weight = i+1
            d2_stage = [
                weigth1d(N=num_feats_to_weight, C=F*C if num_feats_to_weight>1 else 1, 
                         requires_grad=num_feats_to_weight>1),
                to2d(f, c),
                nn.Conv2d(c, int(stride*c), kernel_size=(stride,1), stride=(stride,1), padding=0)
            ]
            
            c = int(stride * c)
            f = int(f // stride)

            d2_stage.extend(
                [ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3,3)],
                    Gdiv=group_divisor, padding='same', activation='gelu')
                    for _ in range(num_blocks)]
                + [to1d(), TimeContextBlock1d(C*F,hC=(C*F)//att_block_red),]
            )

            setattr(self, f'd2_stage_{i+1}', nn.Sequential(*d2_stage))

        num_feats_to_weight_fin = len(stages_setup)+1
        self.fin_wght1d = weigth1d(N=num_feats_to_weight_fin, C=F*C, 
                         requires_grad=num_feats_to_weight>1)


        ######### Speaker embedding layer #########
        if out_channels is None:
            out_channels = C*F

        self.mfa = nn.Sequential(
                nn.Conv1d(F * C, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels, affine=True)
        )
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)
        
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)


    def forward(self, x, label=None):
        # input shape: [B, T]
        with torch.no_grad():
            x = self.stft(x) #  (B, 2, F, T) 

        down_x = {}
        # ====================== Frontend ====================== #
        # ++++++++ Downsample Path  ++++++++ #
        down_x[0] = self.conv1(x)
        for i in range(1, self.n_level+1):
            down_x[i] = getattr(self, f'd_res_{i}')(down_x[i-1])

        # ++++++++ Upsample Path  ++++++++ #
        spec = {}
        for k in ['se', 'ne']:
            x = getattr(self, f'bottleneck_{k}')(down_x[self.n_level])
            for i in range(self.n_level, 0, -1):
                x = torch.cat((x, down_x[i]), 1)
                x = getattr(self, f'u_res_{k}_{i}')(x)
            x = torch.cat((x, down_x[0]), 1)
            spec[k] = getattr(self, f'conv2_{k}')(x)
     

        # ====================== Backend ====================== #
        x = torch.view_as_complex(spec['se'].permute(0, 2, 3, 1).contiguous())
        x = self.stfttomel(x)
        x = self.stem3(x.unsqueeze(1))
        outputs_1d = [x]
        for i in range(1, self.n_level_redimnet+1):
            outputs_1d.append(F.dropout(getattr(self, f'd2_stage_{i}')(outputs_1d), training=self.training))
        x = self.fin_wght1d(outputs_1d)
            
        x = self.mfa(x)
        x = self.bn(self.pool(x))
        x = self.linear(x)
	
        return x, spec
            


def MainModel(**kwargs):

    return DDSE_ReDimNet(**kwargs)


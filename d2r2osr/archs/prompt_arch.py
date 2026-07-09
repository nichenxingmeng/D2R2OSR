## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time

from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp, DCNv2Pack

##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size=96,lin_dim=192, temperature=0.8):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(
            torch.empty(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        trunc_normal_(self.prompt_param, std=0.02)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(lin_dim, prompt_len, 1)
        )
        self.temperature = temperature

    def forward(self,x):
        B,C,H,W = x.shape
        prompt_weights = self.conv(x)
        prompt_weights = prompt_weights - prompt_weights.amax(dim=1, keepdim=True)
        prompt_weights = F.softmax(prompt_weights / self.temperature, dim=1)
        prompt = prompt_weights.unsqueeze(2) * self.prompt_param
        prompt = prompt.sum(dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear",align_corners=False)
        prompt = self.conv3x3(prompt)

        return prompt
    

class PromptGenBlock_SM(nn.Module):
    def __init__(self,
                 prompt_dim=128,
                 prompt_len=5,
                 prompt_size=96,
                 lin_dim=192,
                 temperature=0.8):
        super().__init__()

        self.prompt_len = prompt_len
        self.prompt_dim = prompt_dim
        self.temperature = temperature

        # Prompt basis
        self.prompt_param = nn.Parameter(
            torch.empty(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        trunc_normal_(self.prompt_param, std=0.02)

        # Spatial weight generator
        self.weight_gen = nn.Sequential(
            nn.Conv2d(lin_dim, lin_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(lin_dim // 4, prompt_len, 1)
        )

        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

        self.norm = nn.GroupNorm(1, prompt_dim)  # 稳定训练

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        B, C, H, W = x.shape

        # ===== Spatial weights =====
        prompt_weights = self.weight_gen(x)  # (B, L, H, W)

        prompt_weights = prompt_weights - prompt_weights.amax(dim=1, keepdim=True)
        prompt_weights = F.softmax(prompt_weights / self.temperature, dim=1)

        # ===== Resize prompt basis to feature resolution =====
        prompt_basis = self.prompt_param  # (1, L, Pdim, Psize, Psize)

        if prompt_basis.shape[-1] != W or prompt_basis.shape[-2] != H:
            prompt_basis = F.interpolate(
                prompt_basis.view(-1, self.prompt_dim,
                                  prompt_basis.shape[-1],
                                  prompt_basis.shape[-1]),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            prompt_basis = prompt_basis.view(
                1, self.prompt_len, self.prompt_dim, H, W
            )

        # ===== Spatial mixture =====
        prompt_weights = prompt_weights.unsqueeze(2)  # (B, L, 1, H, W)

        prompt = prompt_weights * prompt_basis  # broadcast
        prompt = prompt.sum(dim=1)  # (B, Pdim, H, W)

        prompt = self.conv3x3(prompt)
        prompt = self.norm(prompt)

        return prompt


class PromptGenBlock_DCN_SM(nn.Module):
    """
    DCN + Spatial Mixture Prompt Generator
    """

    def __init__(self,
                 prompt_dim=128,
                 prompt_len=5,
                 prompt_size=96,
                 lin_dim=192,
                 temperature=0.8):

        super(PromptGenBlock_DCN_SM, self).__init__()

        self.prompt_len = prompt_len
        self.prompt_dim = prompt_dim
        self.temperature = temperature

        self.prompt_param = nn.Parameter(
            torch.empty(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        trunc_normal_(self.prompt_param, std=0.02)

        self.dcn = DCNv2Pack(lin_dim, lin_dim, 3, padding=1)

        self.offset_conv = nn.Sequential(
            nn.Conv2d(1, lin_dim, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(lin_dim, lin_dim, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.weight_conv = nn.Sequential(
            nn.Conv2d(lin_dim, lin_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(lin_dim, prompt_len, 1)
        )

        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

    def forward(self, x, condition):
        B, C, H, W = x.shape
        L = self.prompt_len
        D = self.prompt_dim

        # ========= DCN =========
        offset = torch.tanh(self.offset_conv(condition))
        feat = self.dcn(x, offset)

        weight = self.weight_conv(feat)  # (B, L, H, W)
        weight = weight - weight.amax(dim=1, keepdim=True)
        weight = F.softmax(weight / self.temperature, dim=1)

        # (1, L, D, Hp, Wp)
        prompt_basis = self.prompt_param

        # expand batch
        prompt_basis = prompt_basis.expand(B, -1, -1, -1, -1)

        # reshape 成 4D 才能插值
        prompt_basis = prompt_basis.reshape(
            B * L,
            D,
            prompt_basis.shape[-2],
            prompt_basis.shape[-1]
        )

        # interpolate
        prompt_basis = F.interpolate(
            prompt_basis,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        # reshape 回 5D
        prompt_basis = prompt_basis.view(
            B,
            L,
            D,
            H,
            W
        )

        # ========= Spatial mixture =========
        weight = weight.unsqueeze(2)  # (B,L,1,H,W)
        prompt = weight * prompt_basis
        prompt = prompt.sum(dim=1)  # (B,D,H,W)

        prompt = self.conv3x3(prompt)

        return prompt


class PromptGenBlock_DCN(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192, temperature=0.8):
        super(PromptGenBlock_DCN,self).__init__()
        self.prompt_param = nn.Parameter(
            torch.empty(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        trunc_normal_(self.prompt_param, std=0.02)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.dcn = DCNv2Pack(lin_dim, lin_dim, 3, padding=1)
        self.offset_conv = nn.Sequential(nn.Conv2d(1, lin_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(lin_dim, lin_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(lin_dim, prompt_len, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.temperature = temperature
        
    def forward(self, x, condition):
        B,C,H,W = x.shape
        prompt = self.dcn(x , torch.tanh(self.offset_conv(condition)))
        prompt_weights = self.conv(prompt)
        prompt_weights = prompt_weights - prompt_weights.amax(dim=1, keepdim=True)
        prompt_weights = F.softmax(prompt_weights / self.temperature, dim=1)
        prompt = prompt_weights.unsqueeze(2) * self.prompt_param
        prompt = prompt.sum(dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear",align_corners=False)
        prompt = self.conv3x3(prompt)

        return prompt
    
class PromptGenBlock_Concat(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock_Concat,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        #self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv = nn.Sequential(*[
            nn.Conv2d(lin_dim,prompt_len,kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ])
        

    def forward(self,x):

        B,C,H,W = x.shape
        #emb = x.mean(dim=(-2,-1))
        #prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt_weights = self.conv(x)
        #prompt_weights = prompt_weights.view(B, self.prompt_len, 16, 16)
        prompt = prompt_weights.unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #prompt = prompt_weights.unsqueeze(2) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #print(prompt.shape)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class PromptGenBlock_DCN_Concat(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock_DCN_Concat,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.dcn = DCNv2Pack(lin_dim, lin_dim, 3, padding=1)
        self.offset_conv = nn.Sequential(nn.Conv2d(1, lin_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(lin_dim, lin_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv = nn.Sequential(*[
            nn.Conv2d(lin_dim,prompt_len,kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ])
        

    def forward(self, x, condition):
        B,C,H,W = x.shape
        #emb = x.mean(dim=(-2,-1))
        #prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = self.dcn(x , self.offset_conv(condition))
        prompt_weights = self.conv(prompt)
        #prompt_weights = prompt_weights.view(B, self.prompt_len, 16, 16)
        prompt = prompt_weights.unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #prompt = prompt_weights.unsqueeze(2) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #print(prompt.shape)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt
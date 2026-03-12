import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from dataclasses import asdict
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from xscene.occ_vae.networks.blocks import TriplaneGroupResnetBlock, BeVplaneGroupResnetBlock, DecoderMLPSkipConcat, ResnetBlock, Attn_ResBlock_hw, TriplaneDeformableAttention
from xscene.occ_vae.utils.utils import compose_featmaps, decompose_featmaps

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down, xy_down, padding_mode, kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.xy_down = xy_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        if self.z_down :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        elif self.xy_down:
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        x = self.conv0(x)  # [b, geo_feat_channels, X, Y, Z]

        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        x = self.downsample(x)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return x  # [b, geo_feat_channels, X//2, Y//2, Z//2]

class Encoder_v2(nn.Module):
    def __init__(self, geo_feat_channels, z_down, xy_down, padding_mode, kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.xy_down = xy_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)

        # block1
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

        # block2
        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        x = self.conv0(x)  # [b, geo_feat_channels, X, Y, Z]

        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        x = self.downsample1(x)  # [b, geo_feat_channels, X//2, Y//2, Z]

        residual_feat = x
        x = self.convblock2(x)  # [b, geo_feat_channels, X//2, Y//2, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        x = self.downsample2(x)  # [b, geo_feat_channels, X//4, Y//4, Z]

        residual_feat = x
        x = self.convblock3(x)
        x = x + residual_feat

        return x  # [b, geo_feat_channels, X//4, Y//4, Z]

from dataclasses import dataclass
@dataclass
class Config(ConfigMixin):
    num_class: int = 18
    geo_feat_channels: int = 16
    feat_channel_up: int = 64
    mlp_hidden_channels: int = 256
    mlp_hidden_layers: int = 4
    padding_mode: str = "replicate"
    z_down: bool = False
    xy_down: bool = True
    xy_down_times: int = 2
    use_vae: bool = False
    voxel_fea: bool = False
    triplane: bool = True
    use_deform_attn: bool = True
    pos: bool = True
    dataset: str = "Occ3D-nuScenes"
    block_out_channels: list = (64, 128, 256)

    def to_dict(self):
        base_dict = asdict(self)
        base_dict["_class_name"] = self.__class__.__name__
        base_dict["_diffusers_version"] = "0.17.1"
        return base_dict

class AutoEncoderGroupSkip(ModelMixin, ConfigMixin):
    def __init__(self, 
            num_class = 18,
            geo_feat_channels = 8,
            feat_channel_up = 64,
            mlp_hidden_channels = 256,
            mlp_hidden_layers = 4,
            padding_mode = "replicate",
            z_down = True,
            xy_down = True,
            xy_down_times = 4,
            use_vae = True,
            voxel_fea = False,
            triplane = True,
            use_deform_attn = True,
            pos = True,
            dataset = "Occ3D-nuScenes",
            block_out_channels = (64, 128, 256)
        ) -> None:
        super().__init__()
        self._config = Config(
            num_class=num_class,
            geo_feat_channels=geo_feat_channels,
            feat_channel_up=feat_channel_up,
            mlp_hidden_channels=mlp_hidden_channels,
            mlp_hidden_layers=mlp_hidden_layers,
            padding_mode=padding_mode,
            z_down=z_down,
            xy_down=xy_down,
            xy_down_times=xy_down_times,
            use_vae=use_vae,
            voxel_fea=voxel_fea,
            triplane=triplane,
            use_deform_attn=use_deform_attn,
            pos=pos,
            dataset=dataset,
            block_out_channels=block_out_channels,
        )
        self.num_class = num_class 
        self.embedding = nn.Embedding(num_class, geo_feat_channels)

        self.voxel_fea = voxel_fea
        self.triplane = triplane
        self.use_deform_attn = use_deform_attn
        self.use_vae = use_vae

        # print('build encoder...')
        if dataset == 'nuScenes-Occupancy':
            self.geo_encoder = Encoder(geo_feat_channels, z_down, xy_down, padding_mode, kernel_size = 3, padding = 1)  # kernel_size=(5, 5, 3), padding=(2, 2, 1)
        else:
            # downsample 2x on xy plane
            if xy_down_times == 2:
                self.geo_encoder = Encoder(geo_feat_channels, z_down, xy_down, padding_mode, kernel_size = 3, padding = 1)
            # downsample 4x on xy plane
            elif xy_down_times == 4:
                self.geo_encoder = Encoder_v2(geo_feat_channels, z_down, xy_down, padding_mode, kernel_size = 3, padding = 1)
            else:
                raise ValueError(f"xy_down_times {xy_down_times} not supported, only 2 or 4 is supported")

        if use_vae:
            mid_geo_feat_channels = geo_feat_channels * 8
            self.mid_block = nn.Sequential(
                ResnetBlock(in_channels=geo_feat_channels, out_channels=mid_geo_feat_channels, temb_channels=0, dropout=0.0),
                Attn_ResBlock_hw(mid_geo_feat_channels),
                ResnetBlock(in_channels=mid_geo_feat_channels, out_channels=mid_geo_feat_channels, temb_channels=0, dropout=0.0)
            )

            self.proj_xy = nn.Sequential(
                nn.GroupNorm(num_groups=mid_geo_feat_channels // 4, num_channels=mid_geo_feat_channels, eps=1e-6, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_geo_feat_channels, 2*geo_feat_channels, kernel_size=3, stride=1, padding=1)
            )
            self.proj_xz = nn.Sequential(
                nn.GroupNorm(num_groups=mid_geo_feat_channels // 4, num_channels=mid_geo_feat_channels, eps=1e-6, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_geo_feat_channels, 2*geo_feat_channels, kernel_size=3, stride=1, padding=1)
            )
            self.proj_yz = nn.Sequential(
                nn.GroupNorm(num_groups=mid_geo_feat_channels // 4, num_channels=mid_geo_feat_channels, eps=1e-6, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_geo_feat_channels, 2*geo_feat_channels, kernel_size=3, stride=1, padding=1)
            )

        if voxel_fea :
            self.norm = nn.InstanceNorm3d(geo_feat_channels) 
        else:
            self.norm = nn.InstanceNorm2d(geo_feat_channels)
        self.geo_feat_dim = geo_feat_channels
        self.pos = pos
        self.pos_num_freq = 6  # the defualt value 6 like NeRF

        if self.use_deform_attn:
            # Instantiate Deformable Attention Module
            # pos_dim = 3 (dims) * 2 (sin/cos) * 6 (freqs) = 36
            self.pos_dim = 3 * 2 * self.pos_num_freq 
            self.deform_attn_module = TriplaneDeformableAttention(pos_dim=self.pos_dim)
        
        # print('triplane features are summed for decoding...')
        if dataset == 'nuScenes-Occupancy':
            if voxel_fea:
                self.geo_convs = nn.Sequential(
                    nn.Conv3d(geo_feat_channels, feat_channel_up, kernel_size=3, stride=1, padding=1, bias=True, padding_mode=padding_mode),
                    nn.InstanceNorm3d(geo_feat_channels)
                )
            else: 
                if triplane:
                    self.geo_convs = TriplaneGroupResnetBlock(geo_feat_channels, feat_channel_up, ks=3, input_norm=False, input_act=False)
                else : 
                    self.geo_convs = BeVplaneGroupResnetBlock(geo_feat_channels, feat_channel_up, ks=3, input_norm=False, input_act=False)
        else:
            self.geo_convs = TriplaneGroupResnetBlock(geo_feat_channels, feat_channel_up, ks=3, input_norm=False, input_act=False)

        # print(f'build shared decoder... (PE: {self.pos})\n')
        if self.pos:
            self.geo_decoder = DecoderMLPSkipConcat(feat_channel_up+6*self.pos_num_freq, num_class, mlp_hidden_channels, mlp_hidden_layers)
        else:
            self.geo_decoder = DecoderMLPSkipConcat(feat_channel_up, num_class, mlp_hidden_channels, mlp_hidden_layers)

    @property
    def config(self):
        return self._config

    @classmethod
    def from_config(cls, config, **kwargs):
        config = dict(config)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        return cls(**config)
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def geo_parameters(self):
        params = list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
        if self.use_deform_attn:
            params += list(self.deform_attn_module.parameters())
        return params
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + list(self.tex_decoder.parameters())

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def encode(self, vol):
        x = vol.detach().clone()
        x[x == 255] = 0
            
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        vol_feat = self.geo_encoder(x)

        if self.voxel_fea:
            vol_feat = self.norm(vol_feat).tanh()
            return vol_feat
        else:
            xy_feat = vol_feat.mean(dim=4)
            xz_feat = vol_feat.mean(dim=3)
            yz_feat = vol_feat.mean(dim=2)
            
            xy_feat = (self.norm(xy_feat) * 0.5).tanh()
            xz_feat = (self.norm(xz_feat) * 0.5).tanh()
            yz_feat = (self.norm(yz_feat) * 0.5).tanh()

            if self.use_vae:
                triplane_feat = compose_featmaps(xy_feat, xz_feat, yz_feat, tri_size=(vol_feat.shape[2], vol_feat.shape[3], vol_feat.shape[4]))[0]
                triplane_feat = self.mid_block(triplane_feat)
                xy_feat, xz_feat, yz_feat = decompose_featmaps(triplane_feat, tri_size=(vol_feat.shape[2], vol_feat.shape[3], vol_feat.shape[4]))

                xy_out = self.proj_xy(xy_feat)
                xy_mean, xy_logvar = torch.chunk(xy_out, 2, dim=1)
                xy_feat = self.reparameterize(xy_mean, xy_logvar)

                xz_out = self.proj_xz(xz_feat)
                xz_mean, xz_logvar = torch.chunk(xz_out, 2, dim=1)
                xz_feat = self.reparameterize(xz_mean, xz_logvar)

                yz_out = self.proj_yz(yz_feat)
                yz_mean, yz_logvar = torch.chunk(yz_out, 2, dim=1)
                yz_feat = self.reparameterize(yz_mean, yz_logvar)
                return (xy_feat, xz_feat, yz_feat), (xy_mean, xz_mean, yz_mean), (xy_logvar, xz_logvar, yz_logvar)
            else:
                return [xy_feat, xz_feat, yz_feat]  # [B,C,X,Y  B,C,X,Z  B,C,Y,Z]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [bs, C, H, W]
        # x: [bs, N, 2]
        sample_coords = x.view(x.shape[0], 1, -1, 2) # sample_coords: [bs, 1, N, 2]
        feat = F.grid_sample(feat_map, sample_coords.flip(-1), align_corners=False, padding_mode='border') # feat : [bs, C, 1, N]
        feat = feat[:, :, 0, :] # feat : [bs, C, N]
        feat = feat.transpose(1, 2) # feat : [bs, N, C]
        return feat

    def sample_feature_plane3D(self, vol_feat, x):
        """Sample feature map at given coordinates"""
        # feat_map: [bs, C, H, W, D]
        # x: [bs, N, 3]
        sample_coords = x.view(x.shape[0], 1, 1, -1, 3)
        feat = F.grid_sample(vol_feat, sample_coords.flip(-1), align_corners=False, padding_mode='border') # feat : [bs, C, 1, 1, N]
        feat = feat[:, :, 0, 0, :] # feat : [bs, C, N]
        feat = feat.transpose(1, 2) # feat : [bs, N, C]
        return feat 

    def decode(self, feat_maps, query):        
        # feat_maps = [B,C,X,Y  B,C,X,Z  B,C,Y,Z]
        # query = B,N,C [-1,1] query_coord
        PE = None
        if self.pos or getattr(self, 'use_deform_attn', False):
            PE_list = []
            for freq in range(self.pos_num_freq):
                PE_list.append(torch.sin((2.**freq) * query))
                PE_list.append(torch.cos((2.**freq) * query))
            PE = torch.cat(PE_list, dim=-1)  # [bs, N, 6*self.pos_num_freq]

        if self.voxel_fea:
            h_geo = self.geo_convs(feat_maps)
            h_geo = self.sample_feature_plane3D(h_geo, query)
        else : 
            # coords [N, 3]
            coords_list = [[0, 1], [0, 2], [1, 2]]
            geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
            geo_feat_maps = self.geo_convs(geo_feat_maps)

            if self.triplane:
                if getattr(self, 'use_deform_attn', False):
                    h_geo = self.deform_attn_module(query, PE, geo_feat_maps)
                else:
                    h_geo = 0
                    for i in range(3):
                        h_geo += self.sample_feature_plane2D(geo_feat_maps[i], query[..., coords_list[i]])
            else :
                h_geo = self.sample_feature_plane2D(geo_feat_maps[0], query[..., coords_list[0]]) # feat : [bs, N, C]
            
        if self.pos :
            h_geo = torch.cat([h_geo, PE], dim=-1)

        h = self.geo_decoder(h_geo) # h : [bs, N, 1]
        return h
    
    def forward(self, vol, query):
        if self.use_vae:
            feat_map, means, logvars = self.encode(vol)
            out = self.decode(feat_map, query)
            return out, means, logvars
        else:
            feat_map = self.encode(vol) # [B,C,X,Y  B,C,X,Z  B,C,Y,Z]
            return self.decode(feat_map, query)

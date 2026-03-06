import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from pytorch3d.ops import knn_gather

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            # Uni3D ##################################
            nn.Conv1d(6, 128, 1),
            ##########################################
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        # Uni3D ##################################
        point_groups = point_groups.reshape(bs * g, n, 6)
        ##########################################
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


# Uni3D ##################################
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def nn_search( self, Q_pts, T_pts, knn ) :
        # Q_pts : an input batch of query 3D points [B, P1, 3]
        # T_pts : an input batch of target 3D point sets [B, P2, 3]

        # The knn_points function in Pytorch3D is slow when knn is very large
        # _, nn_idx, _ = knn_points( Q_pts, T_pts, K=knn ) # [B, P1, K]

        distmats = torch.cdist( Q_pts, T_pts )
        dists, nn_idx = torch.topk( distmats, knn, dim=2, largest=False, sorted=True )
        return nn_idx

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        # idx = knn_point(self.group_size, xyz, center) # B G M
        idx = self.nn_search( center, xyz, self.group_size )
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features
##########################################

# Uni3D ##################################
# From PyTorch internals
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1.bias is not None:
            nn.init.ones_(self.fc1.bias[self.fc1.bias.shape[0] // 2:])
        nn.init.normal_(self.fc1.weight[self.fc1.weight.shape[0] // 2:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class EvaAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim = None,
            norm_layer = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = False # use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope = None,
            attn_mask = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
##########################################



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # Uni3D ##################################
        mlp_hidden_dim = int(dim * (4 * 2 / 3) )
        self.mlp = GluMlp(in_features=dim, hidden_features=mlp_hidden_dim*2, act_layer=nn.SiLU, drop=drop, gate_last=False )
        self.attn = EvaAttention(
            dim, num_heads=num_heads, qkv_bias=True, qkv_fused=True, attn_drop=attn_drop, proj_drop=drop)
        ##########################################

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x




class TransformerWithAdapter(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., stag_size=0):
        super().__init__()

        ################################
        # Create side network
        if( stag_size == 0 ): # STAG-std (standard size)
            print("-----------STAG-std------------")
            self.token_accumulation_modules = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            self.graph_convolution_modules = ["6", "7", "8", "9", "10", "11"]
            rank = 192 # latent dimension
            heads = 1 # number of heads (= number of groups for group convolution)
            self.n_share_tam = 12 # adjacent n_share token accum. modules share parameters
            self.n_share_gcm = 6 # adjacent n_share graph conv. modules share parameters
            self.K = 8 # number of nearest neighbors used for graph conv.
        elif( stag_size == 1 ): # STAG-sl (slightly large)
            print("-----------STAG-sl------------")
            self.token_accumulation_modules = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            self.graph_convolution_modules = ["3", "4", "5", "6", "7", "8", "9", "10", "11"]
            rank = 192 # latent dimension
            heads = 1 # number of heads (= number of groups for group convolution)
            self.n_share_tam = 3 # adjacent n_share token accum. modules share parameters
            self.n_share_gcm = 3 # adjacent n_share graph conv. modules share parameters
            self.K = 8 # number of nearest neighbors used for graph conv.
        else:
            print( "error: undefined stag_size." )
            quit()

        # Create down-projection layers for residual connection
        self.addparam_downproj_dict = nn.ModuleDict({})
        for i in self.token_accumulation_modules:
            module_id = str( int(i) // self.n_share_tam )
            if( not module_id in self.addparam_downproj_dict ):
                self.addparam_downproj_dict[ module_id ] = nn.Sequential( nn.Conv1d( embed_dim, rank, kernel_size=1, groups=heads ) )

        # Create graph convolution layers and up-projection layers for token adaptation
        self.addparam_graphconv1_dict = nn.ModuleDict({})
        self.addparam_graphconv2_dict = nn.ModuleDict({})
        self.addparam_fc_dict = nn.ModuleDict({})
        self.addparam_upproj_dict = nn.ModuleDict({})
        for i in self.graph_convolution_modules:
            init_id = int(self.graph_convolution_modules[0])
            module_id = str( ( int(i) - init_id ) // self.n_share_gcm + init_id )
            if( not module_id in self.addparam_graphconv1_dict ):
                self.addparam_graphconv1_dict[ module_id ] = nn.Sequential( nn.Conv1d( rank, rank, kernel_size=1, groups=heads ) )
                # self.addparam_graphconv1_dict[ module_id ] = nn.Sequential( nn.Conv2d( 2*rank, rank, kernel_size=1, groups=heads ) )
                self.addparam_graphconv2_dict[ module_id ] = nn.Sequential( nn.Conv1d( rank, rank, kernel_size=1, groups=heads ) )
                self.addparam_fc_dict[ module_id ] = nn.Sequential( nn.Conv1d( rank, rank, kernel_size=1, groups=1 ) )
                self.addparam_upproj_dict[ module_id ] = nn.Sequential( nn.Conv1d( rank, embed_dim, kernel_size=1, groups=heads ), nn.ReLU() )
        ################################

        ################################
        # Create Transformer backbone
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])
        ################################

    def graph_conv( self, x, centers, depth ):
        B, C, N = x.shape

        distmats = torch.cdist( centers, centers )
        # distmats = torch.cdist( x.permute(0,2,1), x.permute(0,2,1) )

        _, nn_idx = torch.topk( distmats, self.K, dim=2, largest=False, sorted=True )

        # Efficient EdgeConv
        x_center = self.addparam_graphconv1_dict[str(depth)]( x )
        x_center = x_center.unsqueeze(2)
        x_neighbor = self.addparam_graphconv2_dict[str(depth)]( x )
        x_neighbor = knn_gather( x_neighbor.permute(0,2,1), nn_idx ).permute(0,3,2,1)
        x = x_center + x_neighbor
        x = x.max(2)[0]
        x = self.addparam_fc_dict[str(depth)]( x )

        # # EdgeConv
        # x_center = x.unsqueeze(2).repeat(1,1,self.K,1)
        # x_neighbor = knn_gather( x.permute(0,2,1), nn_idx ).permute(0,3,2,1)
        # x = torch.cat( [ x_center, x_neighbor ], dim=1 )
        # x = self.addparam_graphconv1_dict[str(depth)]( x )
        # x = x.max(2)[0]
        # x = self.addparam_fc_dict[str(depth)]( x )

        # # simple graphconv
        # x = self.addparam_graphconv1_dict[str(depth)]( x )
        # x = knn_gather( x.permute(0,2,1), nn_idx ).permute(0,3,2,1)
        # x = x.max(2)[0]
        # x = self.addparam_fc_dict[str(depth)]( x )

        # # local max-pooling
        # x = knn_gather( x.permute(0,2,1), nn_idx ).permute(0,3,2,1)
        # x = x.max(2)[0]
        # x = self.addparam_fc_dict[str(depth)]( x )

        return x

    def forward( self, x, pos, center ):

        # Uni3D ##################################
        x = x + pos
        ##########################################

        residuals = 0.0
        for i, block in enumerate(self.blocks):

            # Token accumulation
            if( str(i) in self.token_accumulation_modules ):
                module_id = int(i) // self.n_share_tam
                residuals = residuals + self.addparam_downproj_dict[str( module_id )](x[ :, 1:, : ].permute(0,2,1)) # remove cls token before down-projection

            # Uni3D ##################################
            x = block( x )
            ##########################################

            # Token adaptation via graph convolution
            if( str(i) in self.graph_convolution_modules ):

                init_id = int(self.graph_convolution_modules[0])
                module_id = ( int(i) - init_id ) // self.n_share_gcm + init_id

                residuals = self.graph_conv( residuals, center, module_id )

                # compute displacements for tokens
                d = self.addparam_upproj_dict[str( module_id )]( residuals ).permute(0,2,1)
                d = torch.cat( [ d.mean(1).unsqueeze(1), d ], dim=1 ) # displacement for cls token is computed by average pooling

                x = x + d # token adaptation

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, pts, vis = False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B,_,C = x_vis.shape # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # Uni3D ##################################
        self.encoder2trans = nn.Linear(self.encoder_dims,  self.trans_dim)
        ##########################################

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        # self.blocks = TransformerEncoder(
        #     embed_dim=self.trans_dim,
        #     depth=self.depth,
        #     drop_path_rate=dpr,
        #     num_heads=self.num_heads,
        # )
        self.blocks = TransformerWithAdapter(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            stag_size=config.stag_size,
        )

        # self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                # nn.Linear(self.trans_dim * 2, 256),
                nn.Linear(self.trans_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)

            # Uni3D ##################################
            base_ckpt = {k.replace("point_encoder.", ""): v for k, v in ckpt['module'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('visual') :
                    base_ckpt[k.replace("visual.", "blocks.") ] = base_ckpt[k]
                    del base_ckpt[k]
            ##########################################

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        # Uni3D ##################################
        colors = 0.4 * torch.ones_like( pts, device=pts.device )
        _, center, features = self.group_divider(pts, colors)
        group_input_tokens = self.encoder(features)  # B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        ##########################################

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos, center)
        # x = self.norm(x)

        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # ret = self.cls_head_finetune(concat_f)

        global_feats = x.mean(1)
        ret = self.cls_head_finetune( global_feats )

        return ret

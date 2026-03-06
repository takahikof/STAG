import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from models.MaskLRF import PointTransformer

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def get_k_nn(xyz1, xyz2, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)

    dists, inds = dists[:, :, :k], inds[:, :, :k]
    return dists, inds

def interpolate(xyz1, xyz2, feature, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)   N1>N2
    :param feature: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = feature.shape
    dists, inds = get_k_nn(xyz1, xyz2, k)

    # inversed_dists = 1.0 / (dists + 1e-8)
    #
    # weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    #
    # weight = torch.unsqueeze(weight, -1)

    interpolated_feature = gather_points(feature, inds)  # shape=(B, N1, 3, C2)

    # return interpolated_feature, inds, weight
    return interpolated_feature, inds

# ref: PaRot: Patch-Wise Rotation-Invariant Network via Feature Disentanglement and Pose Restoration
# based on: https://github.com/dingxin-zhang/PaRot
class FP_Module_angle(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_angle, self).__init__()

        dim_posembed = 32
        self.posembed = nn.Sequential(
            nn.Conv2d( 3+1, dim_posembed, kernel_size=1, bias=False),
            nn.BatchNorm2d( dim_posembed ),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.backbone = nn.Sequential()
        bias = False if bn else True

        in_channels = in_channels + dim_posembed
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz1, xyz2, feat2, lrf2, k=3):

        B, N1, _ = xyz1.shape
        _, N2, C2 = feat2.shape

        interpolated_feature, inds = interpolate(xyz1, xyz2, feat2, k) # get features of neighboring points

        lrf2 = lrf2.reshape( B, N2, 9 )
        close_lrf = gather_points( xyz2, inds )
        lrf2 = gather_points( lrf2, inds ).view(-1, 3, 3)

        relate_position = xyz1.unsqueeze(2).repeat(1, 1, k, 1) - close_lrf

        for_dot = F.normalize(relate_position.view(-1, 3), dim=-1).unsqueeze(2)
        angle = lrf2.matmul(for_dot)
        angle = angle.view(B, N1, k, -1)

        relative_pos = torch.cat((torch.norm(relate_position, dim=-1, keepdim=True), angle), dim=3)
        pos = self.posembed(relative_pos.permute(0, 3, 2, 1))
        interpolated_feature = interpolated_feature.permute(0, 3, 2, 1)
        cat_interpolated_points = torch.cat((interpolated_feature, pos), dim=1)

        new_points = self.backbone(cat_interpolated_points)
        new_points = torch.sum(new_points, dim=2)

        return new_points

# finetune model
@MODELS.register_module()
class PointTransformerForSegmentation(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.cls_dim = config.cls_dim

        self.MAE_encoder = PointTransformer( config, **kwargs )

        num_cls = 16
        dim_global_feat = 256
        dim_token_feat = 256
        dim_cls_label_feat = 64
        dim_prop_feat = 1024
        self.partseg_gf_embedder = nn.Sequential(
            nn.Linear(self.trans_dim * self.config.depth, dim_global_feat),
            nn.BatchNorm1d(dim_global_feat),
            nn.ReLU(inplace=True)
        )
        self.partseg_token_embedder = nn.Sequential(
            nn.Linear(self.trans_dim * self.config.depth, dim_token_feat),
            nn.BatchNorm1d(dim_token_feat),
            nn.ReLU(inplace=True)
        )

        self.partseg_label_conv = nn.Sequential(nn.Conv1d(num_cls, dim_cls_label_feat, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(dim_cls_label_feat),
                                   nn.ReLU(inplace=True))

        self.partseg_propagation = FP_Module_angle( in_channels=dim_token_feat,
                                                    mlp=[ dim_token_feat * 2, dim_prop_feat ] )

        self.partseg_conv1 = nn.Conv1d( dim_global_feat+dim_cls_label_feat+dim_prop_feat, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.partseg_conv2 = nn.Conv1d(512, 256, 1)
        self.partseg_conv3 = nn.Conv1d(256, self.cls_dim, 1)
        self.partseg_bn1 = nn.BatchNorm1d(512)
        self.partseg_bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

        self.build_loss_func()

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
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                # if k.startswith("gf_embedder") :
                #     base_ckpt[ "MAE_encoder." + k ] = base_ckpt[ k ]
                #     del base_ckpt[k]
                if k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

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

    def forward( self, pts, cls_label ):

        B, N, C = pts.shape
        num_class = cls_label.shape[2]

        tokens, centers, _, _, lrf = self.MAE_encoder( pts, return_tokens=True )
        tokens = torch.cat( tokens, dim=2 )

        global_feats = tokens.mean(1)
        global_feats = self.partseg_gf_embedder( global_feats )
        global_feats = global_feats.view(B, -1).unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(B, num_class, 1)
        cls_label_feats = self.partseg_label_conv(cls_label_one_hot).repeat(1, 1, N)
        global_feats = torch.cat( [ global_feats, cls_label_feats ], 1 )

        lrf = lrf.permute( 0, 1, 3, 2 ) # After permutation, the axes correspond to the rows of each 3x3 matrix

        tokens = self.partseg_token_embedder( tokens )

        f_level_0 = self.partseg_propagation( pts[:,:,0:3], centers[:,:,0:3],
                                              tokens, lrf )

        x = torch.cat((f_level_0, global_feats), 1)
        x = self.relu(self.partseg_bn1(self.partseg_conv1(x)))
        x = self.dp1(x)
        x = self.relu(self.partseg_bn2(self.partseg_conv2(x)))
        x = self.partseg_conv3(x)
        x = x.permute(0, 2, 1)

        return x

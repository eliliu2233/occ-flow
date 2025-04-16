
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

# from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext( 
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward']) 


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    
    """ CPU version of multi-scale deformable attention.
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape  # 1 x 16384 x 8 x 32
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape  # 1 x 147456 x 8 x 1 x 1 x 3
    # print(value.shape) # [1, 16384, 8, 32]
    value_list = value.split([Z_ * H_ * W_ for Z_, H_, W_ in value_spatial_shapes],
                             dim=1)  # (1 x 4096 x 8 x 32, )
    sampling_grids = 2 * sampling_locations - 1  # ??? 1 x 147456 x 8 x 1 x 1 x 3
    sampling_value_list = []
    # print(value_list.shape)
    # print(value_spatial_shapes) # [4, 64, 64]
    for level, (Z_, H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        # print(value_list[level].flatten(2).shape) # [1, 16384, 256]
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, Z_, H_, W_)  # 
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        # print(sampling_grids.shape) # [2, 16384, 8, 1, 16, 3]
        _, dim_2, _, _, _, _ = sampling_grids.shape # [1, 147456, 8, 1, 1, 3]
        # print(sampling_grids.shape)
        sampling_grid_l_ = sampling_grids.transpose(1, 2).flatten(0, 1).reshape(bs * num_heads, 128, 128, dim_2//128//128, 3).permute(0,3,1,2,4).reshape(bs * num_heads, dim_2, 1, 1, 3).permute(0,2,1,3,4)
        # print(sampling_grid_l_.shape) # [8, 147456, 1, 1, 3]
        # bs*num_heads, embed_dims, num_queries, num_points
        # import pdb
        # pdb.set_trace()
        # print(H_, W_, Z_) # [4, 64, 64]
        # print(value_l_.shape)
        sampling_value_l_ = F.grid_sample(
            value_l_,  # [8, 32, 4, 64, 64]
            sampling_grid_l_, # 8 x 1 x 16384 x 1 x 3
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False) 
        # print(sampling_value_l_.shape) # [8, 32, 147456, 1, 1]
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)  # 8 x 1 x 147456 x 1
    
    # print(torch.stack(sampling_value_list, dim=-3).flatten(-3).shape, attention_weights.shape)
    # [16, 32, 16, 16384] [16, 1, 262144, 1]
    sampling_tmp = torch.stack(sampling_value_list, dim=-3).flatten(-3).permute(0,1,3,2).reshape(
        bs * num_heads, -1, num_queries).unsqueeze(3)  # 8 x 32 x 147456 x 1
    # print(sampling_tmp.shape)
    output = (sampling_tmp *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)  # 1 x 256 x 147456
    return output.transpose(1, 2).contiguous()  # 1 x 147456 x 256


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=5,
                 num_lidars=1,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=1),
                 deformable_attention_lidar=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=1),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.deformable_attention_lidar = build_attention(deformable_attention_lidar)
        self.embed_dims = embed_dims
        self.num_cams = num_cams 
        self.num_lidars = num_lidars
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj_pts = nn.Linear(embed_dims, embed_dims)
        # self.output_proj_mm = nn.Linear(2*embed_dims, embed_dims)
        self.output_proj_mm = nn.Sequential(
            # nn.Conv2d(2*embed_dims, embed_dims, 3, padding=1, bias=False),
            # nn.BatchNorm2d(embed_dims),
            # nn.ReLU(True),
            # nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False),
            # nn.BatchNorm2d(embed_dims),
            # nn.ReLU(True),
            nn.Linear(2*embed_dims, embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=False),
        )
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj_pts, distribution='uniform', bias=0.)
        xavier_init(self.output_proj_mm, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'pts_key', 'pts_value', \
        'query_pos', 'reference_points_cam', 'reference_points_lidar'))
    def forward(self,
                query,
                key,
                value,
                pts_key,
                pts_value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                pts_spatial_shapes=None,
                reference_points_cam=None,
                reference_points_pts=None,
                bev_mask=None,
                bev_mask_pts=None,
                level_start_index=None,
                pts_level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        # if pts_key is None:
        #     # print("here")
        #     pts_key = query
        # if pts_value is None: ## 修改
        #     pts_value = pts_key
        # print(residual) # None
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
            # inp_residual_pts = query
            slots_pts = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        # print(query.size()) # [1, 147456, 256]

        D_lidar = reference_points_cam.size(3)  # 6 x 1 x 16384 x 9 x 2
        self.num_cams, bs, spa_hw, spa_z, _ = reference_points_cam.size()  # 6 x 1 x 16384 x 9 x 2
        # reference_points_cam = reference_points_cam.reshape(self.num_cams, bs, -1, 1, 2)
        D_cam = reference_points_cam.size(3)
        # print(reference_points_cam.shape) # [6, 1, 147456, 1, 2]
        # print(D) # 4 num of reference points in a pillar
        indexes = [[] for _ in range(bs)]
        len_query = []
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):  # bev_mask: 6 x 1 x 147456 x 1, mask_per_img: 1 x 147456 x 1 
                # index_query_per_img = mask_per_img[j].flatten(-2).nonzero().squeeze(-1)  # 2557
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                # index_query_per_img_ = mask_per_img[0].nonzero().squeeze(-1)
                # print(index_query_per_img.shape, index_query_per_img_.shape)
                # print(mask_per_img.shape, index_query_per_img.shape) #
                indexes[j].append(index_query_per_img)
                max_len = max(max_len, len(index_query_per_img))

        # max_len = max([len(each) for each in indexes]) # 33539
        # max_len: max number of reference points of 6 cameras

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])  # 1 x 6 x 33539 x 256
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D_cam, 2])  # 1 x 6 x 33539 x 1 x 2
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):  
                # print(reference_points_per_img.shape) # [1, 147456, 1, 2]
                # print(i)
                index_query_per_img = indexes[j][i]
                # print(indexes[0])
                # print(queries_rebatch[j, i, :len(index_query_per_img)].shape, query[j, index_query_per_img].shape) # torch.Size([21138, 256]) torch.Size([21138, 2, 256])
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        num_cams, l, bs, embed_dims = key.shape
        # print(key.shape)
        # print(reference_points_rebatch)

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # 6 x 5800 x 256
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # 6 x 5800 x 256

        # print(query.view(bs*self.num_lidars, num_query, self.embed_dims).shape) # [2, 262144, 256]
        # print(queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims).shape) # [12, 3751, 256]
        # print(level_start_index) # 0
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D_cam, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index, sensor='Cam').view(bs, self.num_cams, max_len, self.embed_dims) # 1 x 6 x max_len x 256
        # print(queries.shape) # [1, 6, 33539, 256]
        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]  # 1 x 147456 x 256

      
        # print(count)
        # slots = slots.reshape(bs, 128, 128, 9, -1).permute(0, 3, 1, 2, 4).reshape(bs, num_query, -1)
        # bev_mask = bev_mask.permute(0, 1, 3, 2)
        count = bev_mask.sum(-1) > 0
        # print(count.shape) # [6, 1, 147456]
        count = count.permute(1, 2, 0).sum(-1)  # 1 x 147456
        count = torch.clamp(count, min=1.0)  # 1 x 147456
        slots = slots / count[..., None] # 1 x 147456 x 256
        slots = self.output_proj(slots) # 1 x 147456 x 256

    ## =================================================== lidar ============================================ ##
        """ queries_rebatch_pts = query.new_zeros(
            [bs, self.num_lidars, num_query, self.embed_dims])
        reference_points_rebatch_pts = reference_points_pts.new_zeros(
            [bs, self.num_lidar, num_query, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_lidar in enumerate(reference_points_cam):  
                index_query_per_img = indexes[i]
                queries_rebatch_pts[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch_pts[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img] """
        # print(key.shape, pts_key.shape) # [12, 30825, 256] # [1, 20352, 2, 256]
        # print(pts_key.shape) # [1, 16384, 1, 256]
        if  pts_key != None:
            num_lidars, l_lidar, _, _ = pts_key.shape  

            pts_key = pts_key.permute(2, 0, 1, 3).reshape(
                bs * self.num_lidars, l_lidar, self.embed_dims)  # [1, 16384, 1, 256] -> [1, 1, 16384, 256] -> [1, 16384, 256]
            pts_value = pts_value.permute(2, 0, 1, 3).reshape(
                bs * self.num_lidars, l_lidar, self.embed_dims)  # 1 x 16384 x 256
            
            # print(spatial_shapes)
            # print(pts_spatial_shapes) # [4, 64, 64]

            # slect_index = spatial_target_mask.flatten(0).nonzero()[:,0]
            # print(slect_index.shape)
            # random.shuffle(id_list)
            # masked_ratio = 0.90
            # slect_index = id_list[:int((1-masked_ratio)*len(id_list))]
            # queries_pts_rebatch = query[:, slect_index, :]

            # reference_points_pts_flatten = reference_points_pts.reshape(bs, num_query, -1)
            # reference_points_pts_rebatch = reference_points_pts_flatten[:, slect_index, :]
            
            query_lidar = F.interpolate(pts_value.reshape(-1,4,64,64,256).permute(0,4,1,2,3), (9, 128, 128), mode='trilinear')
            query_lidar = query_lidar.flatten(2).permute(0,2,1)
            # print(pts_value.shape) # [1, 16384, 256]
            # print(reference_points_pts.shape) # [1, 1, 16384, 9, 3]
            queries_pts = self.deformable_attention_lidar(query=query_lidar.view(bs*self.num_lidars, num_query, self.embed_dims), key=pts_key, 
                                                value=pts_value,
                                                reference_points=reference_points_pts.view(bs*self.num_lidars, num_query // D_lidar, D_lidar, 3), 
                                                spatial_shapes=pts_spatial_shapes,
                                                level_start_index=pts_level_start_index, sensor='Lidar').view(bs, self.num_lidars, num_query, self.embed_dims)  # 1 x 1 x 147456 x 256
                                                
            # print(queries.shape) # [2, 16384, 256]
            # print(queries.shape) # [2, 6, 3751, 256]
            # print(slots.shape) # [2, 262144, 256]
            slots_pts = queries_pts.reshape(bs*self.num_lidars, num_query, self.embed_dims)  # 1 x 147456 x 256
            slots_pts = slots_pts.reshape(bs, 9, 128, 128, -1).permute(0, 2, 3, 1, 4).reshape(bs, num_query, -1)
            # print(slots_pts)
            # print(bev_mask_pts) 
            """ count_pts = bev_mask_pts.sum(-1) > 0
            count_pts = count_pts.permute(1, 2, 0).sum(-1)
            count_pts = torch.clamp(count_pts, min=1.0)
            """
            # slots_pts = slots_pts / count_pts[..., None]
            # print(slots_pts.shape)
            slots_pts = self.output_proj_pts(slots_pts)  # 1 x 147456 x 256
            # print(self.output_proj_pts.weight.grad)
            # print(slots_pts)
        
        # slots_mm = torch.cat((slots, slots_pts), dim=2) # 1 x 147456 x 512
        # slots_mm = self.output_proj_mm(slots_mm)  # 1 x 147456 x 512
        slots_mm = slots
        
        """ slots_mm_tmp = torch.cat((slots, slots_pts), dim=2)
        # slots_mm_reshaped = slots_mm_tmp.permute(0,2,1).reshape(bs, 512, 200, 200)
        # slots_mm_reshaped = self.output_proj_mm(slots_mm_reshaped)
        # slots_mm = slots_mm_reshaped.reshape(bs, 256, 40000).permute(0,2,1) """
        
        """ slots_pts = pts_value
        slots_mm = torch.cat((slots, slots_pts), dim=2)
        slots_mm = self.output_proj_mm(slots_mm) """
        
        return self.dropout(slots_mm) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step    # 64
        self.embed_dims = embed_dims      # 256
        self.num_levels = num_levels      # 4
        self.num_heads = num_heads        # 8
        self.num_points = num_points      # 2 for cam: sampling offset 2
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.sampling_offsets_lidar = nn.Linear(
            embed_dims, num_heads * num_levels * 1 * 3)  ## ???
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.attention_weights_lidar = nn.Linear(embed_dims,
                                           num_heads * num_levels * 1)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.sampling_offsets_lidar, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
            
        grid_init_lidar = torch.stack([thetas.cos(), thetas.sin(), thetas.cos()], -1)
        grid_init_lidar = (grid_init_lidar /
                     grid_init_lidar.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            3).repeat(1, self.num_levels, 1, 1)
        # for i in range(self.num_points):
        #     grid_init_lidar[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.sampling_offsets_lidar.bias.data = grid_init_lidar.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights_lidar, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                sensor=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape    
        bs, num_value, _ = value.shape    
        # print("===checking in spatial cross attention", query.shape, value.shape) # torch.Size([6, 33399, 256]) torch.Size([6, 5800, 256])
        # print(spatial_shapes, num_value) # [58,100]
        if sensor == 'Cam':
            # print(spatial_shapes[:,0]*spatial_shapes[:,1]) # 5800
            # print(num_value) # 5800
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        else:
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1] * spatial_shapes[:, 2]).sum() == num_value

        value = self.value_proj(value)  # 6 x 5800 x 256
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)  # 6 x 5800 x 8 x 32
        if sensor == 'Cam':
            sampling_offsets = self.sampling_offsets(query).view( # query: [6, max_len, 256]
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)  # 6 x max_len x 8 x 1 x 2 x 2
            attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

            attention_weights = attention_weights.softmax(-1)  
            # print(attention_weights.shape) # [6, max_len, 8, 2]

            attention_weights = attention_weights.view(bs, num_query,
                                                       self.num_heads,
                                                       self.num_levels,
                                                       self.num_points)  # 6 x max_len x 8 x 1 x 2
        else: # sensor == "Lidar"
            # print(query.shape) # [1, 16384, 256]
            # print(num_query) # 262144
            # print(num_query) # 147456
            sampling_offsets = self.sampling_offsets_lidar(query).view( 
                bs, num_query, self.num_heads, self.num_levels, 1, 3)  # 1 x 147456 x 8 x 1 x 1 x 3
            # print(num_query) # image: 3751 lidar:262144
            # print(self.num_points) # 16
            # print(self.attention_weights(query).shape) # [2, 262144, 128]
            attention_weights = self.attention_weights_lidar(query).view(
                    bs, num_query, self.num_heads, self.num_levels * 1)  # 1 x 147456 x 8 x 1

            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                       self.num_heads,
                                                       self.num_levels,
                                                       1)  # 1 x 147456 x 8 x 1 x 1

        if reference_points.shape[-1] == 2: # 6 x max_len x 1 x 2
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # 1 x 2
            # print(offset_normalizer) # [100, 58]
            bs, num_query, num_Z_anchors, xy = reference_points.shape # 6 x max_len x 1 x 2
            reference_points = reference_points[:, :, None, None, None, :, :] # 6 x max_len x 1 x 1 x 1 x 1 x 2
            # print(reference_points)
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]  # 1 x 1 x 1 x 1 x 1 x 2
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape 
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)  # 6 x 3802 x 8 x 1 x 2 x 1 x 2
            # print(sampling_offsets.shape) # [6, mex_len, 8, 1, 2, 1, 2]
            # print(reference_points.shape) # [6, max_len, 1, 1, 1, 1, 2]
            sampling_locations = reference_points + sampling_offsets  # [6, 33399, 8, 1, 2, 1, 2]
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)  #  [6, 33399, 8, 1, 2, 2]
            
        elif reference_points.shape[-1] == 3:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack([spatial_shapes[..., 2], spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # print(offset_normalizer)

            bs, num_query, num_Z_anchors, xyz = reference_points.shape 
            # print(reference_points.shape) # [1, 16384, 9, 3]
            ref_dim1, ref_dim2, ref_dim3, ref_dim4 = reference_points.shape 
            # print(reference_points.shape)
            reference_points = reference_points.reshape(ref_dim1, reference_points.shape[1]*reference_points.shape[2], 1 ,ref_dim4)  # 1 x 147456 x 1 x 3
            reference_points = reference_points[:, :, None, None, None, :, :]  # 1 x 147456 x 1 x 1 x 1 x 1 x 3
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]  # offset_normalizer -> 1 x 1 x 1 x 1 x 1 x 3 
            bs, num_query, num_heads, num_levels, num_all_points, xyz = sampling_offsets.shape  # 1 x 147456 x 8 x 1 x 1 x 3
            # print(sampling_offsets.shape) #  1 x 147456 x 8 x 1 x 1 x 3
            # print(reference_points.shape) #  1 x 147456 x 1 x 1 x 1 x 1 x 3
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, 1, 1, xyz)  # 1 x 147456 x 8 x 1 x 1 x 1 x 3
            sampling_locations = reference_points + sampling_offsets
            # print(sampling_locations.shape)
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xyz = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xyz)  # 1 x 147456 x 8 x 1 x 1 x 3

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 3 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #
        # print(sampling_locations)
        """ if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights) """
        
        if sensor == 'Cam':
            # print(torch.max(sampling_locations))
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step) 
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output # 1 x 147456 x 256

import numpy as np
import torch.nn.functional as F
import torch, torch.distributed as dist, torch.nn as nn, os
from nerfstudio.models.neus_custom import NeuSCustomModelConfig
from nerfstudio.fields.sdf_custom_field import SDFCustomFieldConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from ..base_head import BaseTaskHead
from ..nerfacc_head.ray_sampler import RaySampler
from ..nerfacc_head.img2lidar import Img2LiDAR
import nerfacc
from mmseg.models import HEADS
from mmengine.logging import MMLogger
import math
logger = MMLogger.get_instance('occflow')
from utils.tb_wrapper import WrappedTBWriter
if 'occflow' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('occflow')
else:
    writer = None

@HEADS.register_module()
class NeuSHead(BaseTaskHead):

    def __init__(
            self, 
            roi_aabb,
            resolution=0.4,
            near_plane=0.0,
            far_plane=1e10,
            num_samples=64,
            num_samples_importance=64,
            num_samples_outside=0,
            num_up_sample_steps=4,
            base_variance=64,
            beta_init=0.1,

            beta_max=0.195,
            total_iters=3516 * 11,
            use_numerical_gradients=True,
            numerical_gradients_delta=0.01,
            use_uniform_gradient=False,
            nbr_gradient_points=128*128*16,
            calculate_online=False,
            sample_gradient=False,
            use_compact_2nd_grad=False,
            beta_hand_tune=False,
            beta_learnable=True,

            return_uniform_sdf=False,
            estimate_flow=False,
            return_uniform_3d_flow=False,
            return_max_depth=False,
            return_surface_sdf=False,
            return_second_grad=False,
            return_sample_sdf=False,
            return_sem=False,
            with_sem_free=False,
            disp_sampler=False,

            anneal_aabb=False,
            aabb_every_iters=3516,
            aabb_min_near=10.,
            aabb_min_far_frac=0.25,

            # rays args
            ray_sample_mode='fixed',    # fixed, cellular
            center=False,
            ray_number=[192, 400],      # 192 * 400
            ray_img_size=[768, 1600],
            ray_upper_crop=0,
            ray_x_dsr_max=None,
            ray_y_dsr_max=None,
            # img2lidar args
            trans_kw='img2lidar',
            trans_kw_eval=None,
            novel_view=None,

            # render args
            render_bkgd='white',
            mapping_args=dict(
                nonlinear_mode="linear_upscale",
                h_size=[128, 32],
                h_range=[51.2, 28.8],
                h_half=False,
                w_size=[128, 32],
                w_range=[51.2, 28.8],
                w_half=False,
                d_size=[20, 10],
                d_range=[-4.0, 4.0, 12.0]),

            # mlp decoder
            embed_dims = 128,
            feat_dims = 128,
            color_dims = 0,
            density_layers = 2,
            sh_deg = 2,
            sh_act = "relu",

            init_cfg=None, 
            print_freq=50,
            two_split=True,
            representation='tpv',
            using_2d_img_feats=False,
            **kwargs):
        super().__init__(init_cfg, **kwargs)

        self.ray_sampler = RaySampler(
            ray_sample_mode=ray_sample_mode,
            ray_number=ray_number,
            center=center,
            ray_img_size=ray_img_size,
            ray_upper_crop=ray_upper_crop,
            ray_x_dsr_max=ray_x_dsr_max,
            ray_y_dsr_max=ray_y_dsr_max)
        
        self.ray_sampler_eval = RaySampler(
            ray_sample_mode='fixed',
            ray_number=ray_number,
            ray_img_size=ray_img_size,
            ray_upper_crop=ray_upper_crop)
        
        self.img2lidar = Img2LiDAR(
            trans_kw=trans_kw,
            trans_kw_eval=trans_kw_eval,
            novel_view=novel_view)

        model_config = NeuSCustomModelConfig(
            enable_collider=True,
            near_plane=near_plane,
            far_plane=far_plane,
            background_color=render_bkgd,
            background_model='none',
            scene_contraction_activate=False,
            num_samples=num_samples,
            num_samples_importance=num_samples_importance,
            num_up_sample_steps=num_up_sample_steps,
            num_samples_outside=num_samples_outside,
            base_variance=base_variance,
            beta_hand_tune=beta_hand_tune,
            beta_min=beta_init,
            beta_max=beta_max,
            total_iters=total_iters,
            perturb=True,
            use_lpips=False,

            anneal_aabb=anneal_aabb,
            aabb_every_iters=aabb_every_iters,
            aabb_max_near=near_plane,
            aabb_min_near=aabb_min_near,
            aabb_min_far_frac=aabb_min_far_frac,
            aabb_original=roi_aabb,

            disp_sampler=disp_sampler,
            return_sem=return_sem,

            sdf_field=SDFCustomFieldConfig(
                beta_init = beta_init,
                # beta_hand_tune=beta_hand_tune,
                use_numerical_gradients = use_numerical_gradients,
                mapping_args = mapping_args,
                # mlp decoder
                embed_dims = embed_dims,
                feat_dims = feat_dims,
                color_dims = color_dims,
                density_layers = density_layers,
                sh_deg = sh_deg,
                sh_act = sh_act,
                beta_learnable = beta_learnable,
                second_derivative= return_second_grad,
                representation = representation,
                use_uniform_gradient=use_uniform_gradient,
                nbr_gradient_points=nbr_gradient_points,
                calculate_online=calculate_online,
                sample_gradient=sample_gradient,
                use_compact_2nd_grad=use_compact_2nd_grad,
                using_2d_img_feats=using_2d_img_feats,
                estimate_flow=estimate_flow,
                return_sem=return_sem,
                with_sem_free=with_sem_free,
            ),
        )

        self.model = model_config.setup(
            scene_box=SceneBox(
                aabb=torch.tensor([roi_aabb[:3], roi_aabb[3:]]),
                near=near_plane,
                far=far_plane,
                collider_type='box'
            ),
            num_train_data=0,
        )
        self.model.field.set_numerical_gradients_delta(numerical_gradients_delta)
        self.embed_dims = embed_dims
        self.print_freq = print_freq
        self.resolution = resolution
        self.aabb = roi_aabb
        self.return_uniform_sdf = return_uniform_sdf
        self.return_max_depth = return_max_depth
        self.ray_img_size = ray_img_size
        self.render_size = ray_number
        self.return_uniform_3d_flow = return_uniform_3d_flow
        self.return_surface_sdf = return_surface_sdf
        self.return_second_grad = return_second_grad
        self.return_sample_sdf = return_sample_sdf
        self.return_sem = return_sem

        self.estimate_flow = estimate_flow
        self.z_size = self.model.field.mapping.size_d
        self.bev_size = [self.model.field.mapping.size_h, self.model.field.mapping.size_w]
        self.two_split = two_split
        self.using_2d_img_feats = using_2d_img_feats
        
    def forward_occ(
            self,
            representation,
            metas=None,
            **kwargs):
        
        if isinstance(representation, (tuple, list)):
            device = representation[0].device
        else:
            device = representation.device
        
        if not self.using_2d_img_feats:
            self.model.field.pre_compute_density_color(representation, img_metas=metas)
        else:
            self.model.field.pre_compute_density_color(
                representation, img_feats=kwargs['ms_img_feats'], img_metas=metas)
        aabb = kwargs['aabb'] if 'aabb' in kwargs else self.aabb
        reso = kwargs['resolution'] if 'resolution' in kwargs else self.resolution
        return_flow = self.estimate_flow
        if self.return_sem:
            sdf, sem, sem_logits, xyzs = self.get_uniform_sdf(aabb, reso, device=device)
            result_dict = {'sdf': sdf, 'rep': representation, 'sem': sem, 'logits': sem_logits, 'xyz': xyzs}
        else:
            sdf, xyzs = self.get_uniform_sdf(aabb, reso, device=device)
            result_dict = {'sdf': sdf, 'rep': representation, 'xyz': xyzs}
        if return_flow:
            next_sampled_flow = self.model.field.sample_something(xyzs, self.model.field.volume_pad(self.model.field.fwd_flow))
            result_dict.update(flow=next_sampled_flow.reshape(*xyzs.shape[:-1], 2))
        return result_dict
    
    def get_uniform_sdf(self, aabb, resolution, device, shift=False):
        xs = torch.linspace(
            aabb[0]+resolution/2, aabb[3]-resolution/2, int((aabb[3] - aabb[0]) / resolution), device=device)
        ys = torch.linspace(
            aabb[1]+resolution/2, aabb[4]-resolution/2, int((aabb[4] - aabb[1]) / resolution), device=device)
        zs = torch.linspace(
            aabb[2]+resolution/2, aabb[5]-resolution/2, int((aabb[5] - aabb[2]) / resolution), device=device)
        W, H, D = len(xs), len(ys), len(zs)
        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1).flatten(0, 2)

        if shift:
            random_shift = torch.rand_like(xyzs) * resolution
            xyzs = xyzs + random_shift
        
        if self.return_sem:
            h = self.model.field.forward_geonetwork(xyzs)
            sdf, sem = h[..., 0], h[..., 1:]
            sdf = sdf.reshape(H, W, D)
            sem_logits = sem.reshape(H, W, D, -1)
            sem = torch.argmax(sem, dim=-1).reshape(H, W, D)
            return sdf, sem, sem_logits, xyzs.reshape(H, W, D, -1)
        else:
            sdf = self.model.field.forward_sdfnetwork(xyzs)
            sdf = sdf.reshape(H, W, D)
            return sdf, xyzs.reshape(H, W, D, -1)
    
    def prepare(
            self, 
            representation, 
            metas=None, 
            **kwargs):
        # self.model.field.pre_compute_density_color(representation)# , torch.float)
        if not self.using_2d_img_feats:
            self.model.field.pre_compute_density_color(representation, img_metas=metas)
        else:
            self.model.field.pre_compute_density_color(
                representation, img_feats=kwargs['ms_img_feats'], img_metas=metas)
        return {}

    def render(
            self,
            metas=None,
            batch=0,
            **kwargs):
        amp = 'amp' in os.environ and os.environ['amp'] == 'true'
        with torch.cuda.amp.autocast(enabled=amp):
            if os.environ.get('eval', 'false') == 'true':
                ray_sampler = self.ray_sampler_eval
            else:
                ray_sampler = self.ray_sampler 
            rays = ray_sampler()

            origin, direction = self.img2lidar(metas, rays) # B, N, 3; B, N, R, 3
            bs, num_cams, num_rays = direction.shape[:3]
            assert bs == 1, 'only support bs = 1 currently'
            origin = origin.unsqueeze(2).repeat(1, 1, num_rays, 1).flatten(0, 2)
            direction = direction.flatten(0, 2)
            direction_norm = torch.norm(direction, dim=-1, keepdim=True)
            direction = direction / direction_norm
            frames_id = torch.zeros_like(direction[...,0:1])

            if batch > 0:
                output = {
                    'rgb': [],
                    'acc': [],
                    'depth': [],
                    'weights': [],
                    'ts': [],
                    'deltas': [],
                    'vis_normal': [],
                    'sem': []
                    # 'sdfs': []
                }
                chunks = direction.shape[0] * 1.0 / batch
                chunks = int(math.ceil(chunks))
                origins = torch.chunk(origin, chunks)
                directions = torch.chunk(direction, chunks)
                direction_norms = torch.chunk(direction_norm, chunks)
                for i_chunk in range(chunks):
                    # torch.cuda.empty_cache()
                    ray_bundle = RayBundle(
                        origins=origins[i_chunk],
                        directions=directions[i_chunk],
                        directions_norm=direction_norms[i_chunk],
                        pixel_area=torch.zeros_like(direction_norms[i_chunk]))
                    output_i = self.model(ray_bundle)

                    output['rgb'].append(output_i['rgb'])
                    if self.return_sem:
                        output['sem'].append(output_i['sem'])
                    output['vis_normal'].append(output_i['normal_vis'])
                    output['acc'].append(output_i['accumulation'])
                    output['depth'].append(output_i['depth'])
                    # output['sdfs'].append(output_i['field_outputs'][FieldHeadNames.SDF])
                    # depth_median = output['depth_median'].reshape(bs, num_cams, num_rays)
                    output['weights'].append(output_i['weights'])
                    ray_samples = output_i['ray_samples']
                    # there is bug here before 8.16, did not divide direction_norm
                    ts = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
                    # ts = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                    ts = ts / direction_norms[i_chunk].unsqueeze(1)
                    output['ts'].append(ts)
                    # import pdb; pdb.set_trace()
                    deltas = (ray_samples.frustums.ends - ray_samples.frustums.starts)
                    # deltas = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                    deltas = deltas / direction_norms[i_chunk].unsqueeze(1)
                    output['deltas'].append(deltas)
                
                rgb = torch.cat(output['rgb']).reshape(bs, num_cams, num_rays, -1)
                if self.return_sem:
                    sem = torch.cat(output['sem']).reshape(bs, num_cams, num_rays, -1)
                vis_normal = torch.cat(output['vis_normal']).reshape(bs, num_cams, num_rays, -1)
                acc = torch.cat(output['acc']).reshape(bs, num_cams, num_rays)
                depth = torch.cat(output['depth']).reshape(bs, num_cams, num_rays)
                # sdfs = torch.cat(output['sdfs']).reshape(bs, num_cams, num_rays, -1)
                weights = torch.cat(output['weights']).reshape(bs, num_cams, num_rays, -1, 1)
                ts = torch.cat(output['ts'])
                deltas = torch.cat(output['deltas'])

            else:
                ray_bundle = RayBundle(
                    origins=origin,#.float(),
                    directions=direction,#.float(),
                    directions_norm=direction_norm,#.float(),
                    pixel_area=torch.zeros_like(direction_norm),
                    times=frames_id)#, dtype=torch.float))

                output = self.model(ray_bundle)
                """
                output: dict(
                    "rgb": B, N, R, 3
                    "accumulation": B, N, R, 1
                    "depth": B, N, R, 1
                    "normal": B, N, R, 3
                    "weights": B, N, R, S, 1,
                    "ray_points": B, N, R, S, 3
                    "directions_norm": B, N, R, 1
                    "normal_vis":
                    while training
                    "eik_grad": 
                    "points_norm":
                )
                """
                rgb = output['rgb'].reshape(bs, num_cams, num_rays, -1)
                if self.return_sem:
                    sem = output['sem'].reshape(bs, num_cams, num_rays, -1).softmax(-1)
                vis_normal = output['normal_vis'].reshape(bs, num_cams, num_rays, -1)
                acc = output['accumulation'].reshape(bs, num_cams, num_rays)
                depth = output['depth'].reshape(bs, num_cams, num_rays)
                sdfs = output['field_outputs'][FieldHeadNames.SDF].reshape(bs, num_cams, num_rays, -1)
                # depth_median = output['depth_median'].reshape(bs, num_cams, num_rays)
                weights = output['weights'].reshape(bs, num_cams, num_rays, -1, 1)
                num_samples_per_ray = weights.shape[-2]
                w = weights[0,0,...,0]
                indices = w.max(dim=1).indices
                weight_i = torch.vstack([w[torch.arange(0,w.shape[0]),(indices+i).clamp(0,num_samples_per_ray-1)] for i in range(-5,5)]).sum(0)
                ray_samples = output['ray_samples']
                # there is bug here before 8.16, did not divide direction_norm
                ts = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
                # ts = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                ts = ts / direction_norm.unsqueeze(1)
                # import pdb; pdb.set_trace()
                deltas = (ray_samples.frustums.ends - ray_samples.frustums.starts)
                # deltas = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                deltas = deltas / direction_norm.unsqueeze(1)
                if self.estimate_flow:
                    positions = ray_samples.frustums.get_positions().reshape(bs, -1, num_samples_per_ray, 3) # B * N * R, S, 3
                    positions = positions[:,:num_cams*num_rays].reshape(-1,3)
                    lidar2currImg = torch.from_numpy(np.stack([np.linalg.inv(metas[i]['temImg2lidar']) for i in range(len(metas))])).float().to(positions.device)
                    lidar2temImg = torch.from_numpy(np.stack([metas[i]['lidar2nextImg'] for i in range(len(metas))])).float().to(positions.device)
                    surface_points = output['surface_points'].reshape(-1,3) #
                    next_sampled_flow = self.model.field.sample_something(surface_points, self.model.field.volume_pad(self.model.field.fwd_flow))
                    surface_points = (surface_points + torch.cat([next_sampled_flow,torch.zeros_like(next_sampled_flow[...,0:1])],dim=-1)).reshape(bs, num_cams, num_rays, 3)
                    next_pixel = (lidar2temImg[:,:,:3,:3].reshape(bs,num_cams,1,3,3) @ surface_points.unsqueeze(-1) + lidar2temImg[:,:,:3,3:].reshape(bs,num_cams,1,3,1)).squeeze(-1)
                    next_pixel = next_pixel[...,0:2] / torch.where(next_pixel[..., 2:3]>-5.0, next_pixel[..., 2:3], torch.ones_like(next_pixel[..., :1]) * 1e-7)
                    flow_2d = next_pixel - rays.reshape(1,1,-1,2)
                    curr_pixel = (lidar2currImg[:,:,:3,:3].reshape(bs,num_cams,1,3,3) @ surface_points.unsqueeze(-1) + lidar2currImg[:,:,:3,3:].reshape(bs,num_cams,1,3,1)).squeeze(-1)
                    curr_pixel = curr_pixel[...,0:2] / torch.where(curr_pixel[..., 2:3]>-5.0, curr_pixel[..., 2:3], torch.ones_like(curr_pixel[..., :1]) * 1e-7)
                    scene_flow_2d = curr_pixel - rays.reshape(1,1,-1,2)
                    
                    surface_points = output['surface_points'].reshape(bs, num_cams, num_rays, 3)
                    next_pixel = (lidar2temImg[:,:,:3,:3].reshape(bs,num_cams,1,3,3) @ surface_points.unsqueeze(-1) + lidar2temImg[:,:,:3,3:].reshape(bs,num_cams,1,3,1)).squeeze(-1)
                    next_pixel = next_pixel[...,0:2] / torch.maximum(torch.ones_like(next_pixel[..., :1]) * 1e-5, next_pixel[..., 2:3])
                    static_flow = next_pixel - rays.reshape(1,1,-1,2)

        if self.return_max_depth:
            eps = torch.finfo(deltas.dtype).eps
            deltas_ = deltas.reshape(bs, num_cams, num_rays, -1).cpu()
            weights_ = weights.reshape(bs, num_cams, num_rays, -1).cpu() # .clone()
            weights_[deltas_ < eps] = 0.
            w_per_d = weights_ / deltas_.clamp_min(eps)
            indices = w_per_d.argmax(dim=-1, keepdim=True) # bs, num_cams, num_rays, 1
            max_depth = ts.reshape(bs, num_cams, num_rays, -1).cpu()
            max_depth = torch.gather(max_depth, -1, indices).squeeze(-1).to(rgb.device)

        outputs = {
            'ms_depths': [depth],
            # 'ms_depths_median': [depth_median],
            'ms_colors': [rgb],
            'vis_normal': [vis_normal],
            'ms_accs': [acc],
            'ms_rays': rays,
            'weights_i':weight_i,
            'surface_points':output['surface_points']}
        if self.return_max_depth:
            outputs.update({
                'ms_max_depths': [max_depth]})
        if self.return_sem:
            outputs.update({'sem': [sem]})
        if self.estimate_flow:
            outputs.update({'flow': [flow_2d], 'scene_flow':[scene_flow_2d], 'static_flow':[static_flow]})
        return outputs

    def forward(
            self, 
            representation, 
            metas=None, 
            **kwargs):
        global_iter = kwargs.get('global_iter', None)
        estimate_flow = self.estimate_flow and self.training
        if not kwargs.get('prepare', False):
            if not self.using_2d_img_feats:
                self.model.field.pre_compute_density_color(representation, img_metas=metas)
            else:
                self.model.field.pre_compute_density_color(
                    representation, img_feats=kwargs['ms_img_feats'], img_metas=metas)

        amp = 'amp' in os.environ and os.environ['amp'] == 'true'
        with torch.cuda.amp.autocast(enabled=amp):
            if os.environ.get('eval', 'false') == 'true':
                ray_sampler = self.ray_sampler_eval
            else:
                ray_sampler = self.ray_sampler 
            rays = ray_sampler()

            origin, direction = self.img2lidar(metas, rays) # B, N, 3; B, N, R, 3
            bs, num_cams, num_rays = direction.shape[:3]
            assert bs == 1, 'only support bs = 1 currently'
            origin = origin.unsqueeze(2).repeat(1, 1, num_rays, 1).flatten(1, 2)
            direction = direction.flatten(1, 2)
            direction_norm = torch.norm(direction, dim=-1, keepdim=True)
            direction = direction / direction_norm
            if kwargs['lidar_rays'] != None:
                lidar_origin = kwargs['lidar_rays'][...,:3]
                lidar_direction = kwargs['lidar_rays'][...,3:6]
                lidar_direction_norm = torch.norm(lidar_direction, dim=-1, keepdim=True)
                origin = torch.cat((origin, lidar_origin), dim=1)
                direction = torch.cat((direction, lidar_direction), dim=1)
                direction_norm = torch.cat((direction_norm, lidar_direction_norm), dim=1)
            frames_id = torch.zeros_like(direction[...,0:1])
            origin, direction, direction_norm = origin.reshape(-1,3), direction.reshape(-1,3), direction_norm.reshape(-1,1)
            frames_id = frames_id.reshape(-1,1)

            ray_bundle = RayBundle(
                origins=origin,#.float(),
                directions=direction,#.float(),
                directions_norm=direction_norm,#.float(),
                pixel_area=torch.zeros_like(direction_norm),
                times=frames_id)#, dtype=torch.float)

            output = self.model(ray_bundle, iter=global_iter)
            uniform_sdf = None
            if self.return_uniform_sdf:
                uniform_data = self.get_uniform_sdf(self.aabb, self.resolution, rays.device, True)
                if self.return_sem:
                    uniform_sdf, uniform_sem, _, uniform_xyz = uniform_data
                else:
                    uniform_sdf, uniform_xyz = uniform_data
        if kwargs['lidar_rays'] != None:
            output['depth'] = output['depth'].reshape(bs, -1)
            all_rays = output['depth'].size(1)
            depth, lidar_range = output['depth'][:,:num_cams*num_rays].reshape(bs, num_cams, num_rays), output['depth'][:,num_cams*num_rays:]
            rgb = output['rgb'].reshape(bs, all_rays, -1)[:,:num_cams*num_rays,:].reshape(bs, num_cams, num_rays, -1)
            acc = output['accumulation'].reshape(bs, all_rays)[:,:num_cams*num_rays].reshape(bs, num_cams, num_rays)
            fars = output['fars'].reshape(bs, all_rays)[:,:num_cams*num_rays].reshape(bs, num_cams, num_rays)
            all_weights = output['weights'].reshape(bs, all_rays,-1,1)
            weights, lidar_weights = all_weights[:,:num_cams*num_rays,...].reshape(bs, num_cams, num_rays, -1, 1).float(), all_weights[:,num_cams*num_rays:,...].float()
        else:    
            rgb = output['rgb'].reshape(bs, num_cams, num_rays, -1)
            acc = output['accumulation'].reshape(bs, num_cams, num_rays)
            depth = output['depth'].reshape(bs, num_cams, num_rays)
            fars = output['fars'].reshape(bs, num_cams, num_rays)
            weights = output['weights'].reshape(bs, num_cams, num_rays, -1, 1)
        if self.return_surface_sdf:
            surface_sdf = self.model.field.forward_sdfnetwork(output['surface_points'])
            surface_sdf = surface_sdf.reshape(bs, num_cams, num_rays)
        if self.return_sample_sdf:
            sample_sdf = output['field_outputs'][FieldHeadNames.SDF]
            sample_sdf = sample_sdf[:num_cams*num_rays].reshape(bs, num_cams, num_rays, -1)
        if self.return_sem:
            sem = output['sem'][:num_cams*num_rays].reshape(bs, num_cams, num_rays, -1)
        # depth_median = output['depth_median'].reshape(bs, num_cams, num_rays)
        num_samples_per_ray = weights.shape[-2]
        ray_samples = output['ray_samples']
        # there is bug here before 8.16, did not divide direction_norm
        ts = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        # ts = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
        ts = ts / direction_norm.unsqueeze(1)
        # import pdb; pdb.set_trace()
        deltas = (ray_samples.frustums.ends - ray_samples.frustums.starts)
        # deltas = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
        deltas = deltas / direction_norm.unsqueeze(1)
        if kwargs['lidar_rays'] != None:
            all_ts = ts.reshape(bs, all_rays, -1, 1)
            lidar_ts = all_ts[:,num_cams*num_rays:,...]
            ts = all_ts[:,:num_cams*num_rays,...].flatten(0,1)
            all_deltas = deltas.reshape(bs, all_rays, -1, 1)
            lidar_deltas = all_deltas[:,num_cams*num_rays:,...]
            deltas = all_deltas[:,:num_cams*num_rays,...].flatten(0,1)

        if self.return_max_depth:
            eps = torch.finfo(deltas.dtype).eps
            deltas_ = deltas.reshape(bs, num_cams, num_rays, -1)
            weights_ = weights.reshape(bs, num_cams, num_rays, -1).clone()
            weights_[deltas_ < eps] = 0.
            w_per_d = weights_ / deltas_.clamp_min(eps)
            indices = w_per_d.argmax(dim=-1, keepdim=True) # bs, num_cams, num_rays, 1
            max_depth = ts.reshape(bs, num_cams, num_rays, -1)
            max_depth = torch.gather(max_depth, -1, indices).squeeze(-1)
        
        if estimate_flow:
            positions = ray_samples.frustums.get_positions().reshape(bs, -1, num_samples_per_ray, 3) # B * N * R, S, 3
            positions = positions[:,:num_cams*num_rays].reshape(-1,3)
            # prev_sampled_flow = self.model.field.sample_something(positions, self.model.field.volume_pad(self.model.field.bwd_flow))
            next_sampled_flow = self.model.field.sample_something(positions, self.model.field.volume_pad(self.model.field.fwd_flow))
            next_warp = positions.reshape(bs, num_cams, num_rays, num_samples_per_ray, 3)
            # return render scene flow, stop the gradient of sdf field
            next_sampled_flow = next_sampled_flow.reshape(bs, num_cams, num_rays, num_samples_per_ray, 2)
            scene_flow = torch.sum(weights.detach()*next_sampled_flow, dim=-2)
            # return static optical flow with render depth
            with torch.no_grad():
                lidar2temImg = torch.from_numpy(np.stack([metas[i]['lidar2nextImg'] for i in range(len(metas))])).to(positions.device)
                surface_points = output['surface_points'].reshape(bs, -1, 3).to(lidar2temImg.dtype)
                surface_points = surface_points[:,:num_cams*num_rays].reshape(bs, num_cams, num_rays, 3)
                next_pixel = (lidar2temImg[:,:,:3,:3].reshape(bs,num_cams,1,3,3) @ surface_points.unsqueeze(-1) + lidar2temImg[:,:,:3,3:].reshape(bs,num_cams,1,3,1)).squeeze(-1)
                next_pixel = next_pixel[...,0:2] / torch.maximum(torch.ones_like(next_pixel[..., :1]) * 1e-5, next_pixel[..., 2:3])
                static_flow = next_pixel - rays.reshape(1,1,-1,2)

        if global_iter is not None and \
            global_iter % self.print_freq == 0 and \
            ((not dist.is_initialized()) or dist.get_rank() == 0):
            curr_s = output['inv_s']
            logger.info(f'global iter {global_iter} s: {curr_s}, deltas=0: {(deltas == 0).sum().item()}')
            writer.add_scalar('inv_s', curr_s, global_iter)
        
        weights_for_cams = chunk_cams(weights, num_cams)
        ts_for_cams = chunk_cams(ts, num_cams)
        deltas_for_cams = chunk_cams(deltas, num_cams)
        ray_idx_for_cams = [
            torch.arange(num_rays, device=rgb.device).unsqueeze(-1).repeat(1, num_samples_per_ray).flatten()] * num_cams
        eik_grad = output['eik_grad']
        if estimate_flow:
            # prev_warp_for_cams = chunk_cams(prev_warp, num_cams)
            next_warp_for_cams = chunk_cams(next_warp, num_cams)
            next_flow_for_cams = chunk_cams(next_sampled_flow, num_cams)
        if self.return_sample_sdf:
            sample_sdf_for_cams = chunk_cams(sample_sdf, num_cams)

        if self.two_split and self.img2lidar.two_split:
            depth = depth[:, :(num_cams//2), :]
            # depth_median = depth_median[:, :(num_cams//2), :]
            rgb = rgb[:, (num_cams//2):, ...]
            acc = acc[:, :(num_cams//2), :]
            fars = fars[:, :(num_cams//2), :]
            ray_idx_for_cams = ray_idx_for_cams[:(num_cams // 2)]
            weights_for_cams = weights_for_cams[:(num_cams // 2)]
            ts_for_cams = ts_for_cams[:(num_cams // 2)]
            deltas_for_cams = deltas_for_cams[:(num_cams // 2)]
            if estimate_flow:
                # prev_warp_for_cams = prev_warp_for_cams[:(num_cams // 2)]
                next_warp_for_cams = next_warp_for_cams[:(num_cams // 2)]
            if self.return_max_depth:
                max_depth = max_depth[:, :(num_cams//2), :]
            if self.return_sample_sdf:
                sample_sdf_for_cams = sample_sdf_for_cams[:(num_cams // 2)]
            if self.return_sem:
                sem = sem[:, (num_cams//2):, ...]
        
        outputs = {
            'ms_depths': [depth],
            # 'ms_depths_median': [depth_median],
            'ms_colors': [rgb],
            'ms_accs': [acc],
            'ms_fars': [fars],
            'ms_rays': rays,
            'origin': origin,
            'direction': direction,
            'direction_norm': direction_norm,
            'ray_indices': ray_idx_for_cams,
            'weights': weights_for_cams,
            'ts': ts_for_cams,
            'deltas': deltas_for_cams,
            'eik_grad': eik_grad,
            'uniform_sdf': uniform_sdf,}
        outputs.update(surface_points=output['surface_points'].reshape(bs, -1, 3)[:,:num_cams*num_rays].reshape(bs, num_cams, num_rays, 3))
        if kwargs['lidar_rays'] != None:
            outputs.update(pred_ranges=lidar_range, ts_for_lidar=lidar_ts, weights_for_lidar=lidar_weights)
        if estimate_flow:
            outputs.update({
                # 'prev_warps': prev_warp_for_cams,
                # 'next_warps': next_warp_for_cams,
                'points_warps': next_warp_for_cams,
                'sampled_flows': next_flow_for_cams,
                'scene_flows': scene_flow,
                'static_flows': static_flow,
            })
        if self.return_uniform_3d_flow and estimate_flow:
            sample_flow = self.model.field.fwd_flow
            sample_occ = self.model.field.get_occupancy(self.model.field.density_color[:,0,...])
            outputs.update({'uniform_flow_3d': sample_flow.permute(0,2,3,4,1), 'uniform_occ_3d':sample_occ})
        if self.return_max_depth:
            outputs.update({
                'ms_max_depths': [max_depth]
            })
        if self.return_surface_sdf:
            outputs.update({
                'surface_sdf': surface_sdf
            })
        if self.return_second_grad:
            outputs.update({
                'second_grad': output['field_outputs']['second_grad']
            })
        if self.return_sample_sdf:
            outputs.update({
                'sample_sdf': sample_sdf_for_cams
            })
        if self.return_sem:
            outputs.update({'sem': [sem]})
        return outputs
    

def chunk_cams(tensor, num_cams):
    tensor_for_cams = torch.chunk(
        tensor.reshape(num_cams, -1),
        num_cams, dim=0)
    tensor_for_cams = [t.squeeze() for t in tensor_for_cams]
    return tensor_for_cams
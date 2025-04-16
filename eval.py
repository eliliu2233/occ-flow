import time, argparse, os.path as osp
import os, math, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,6,7"
os.environ["NCCL_P2P_DISABLE"] = "1"
import torch, numpy as np
import torch.distributed as dist
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
from utils.vis_utils import save_surroundimg, semantic_colors

import warnings
warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from utils.feat_tools import multi2single_scale
from copy import deepcopy
from utils.config_tools import modify_for_eval

def downsample_depth(depth, downsample):
    from torchvision.transforms import Resize
    H, W = depth.shape
    depth = depth.reshape(H // downsample, downsample, W // downsample, downsample)
    depth = np.transpose(depth, (0, 2, 1, 3))
    depth = depth.reshape(-1, downsample * downsample)
    depth_tmp = np.where(depth == 0.0, 1e5 * np.ones_like(depth), depth)
    depth = np.min(depth_tmp, axis=-1)
    depth = depth.reshape(H // downsample, W // downsample)
    depth[depth>1e3] = 0.0
    return depth

def depth2rgbimg(depth_img, img_size, max_depth=None, mode='bgr'):
    import cv2
    width, height = img_size[0], img_size[1]
    mask = np.logical_and(depth_img>0, depth_img<80)
    disp_img = np.zeros_like(depth_img)
    disp_img[mask] = 1.0 / depth_img[mask]
    if mask.sum()>0:
        # disp_min = disp_img[mask].min()
        disp_min = 1.0 / 120.0
        # disp_min = 1.0 / 6.0
        disp_max = np.percentile(disp_img[mask], 95)
        disp_img[mask] = 255.0*(disp_img[mask]-disp_min)/(disp_max-disp_min)
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_img.reshape(width,height,1), alpha=1), cv2.COLORMAP_MAGMA) #JET
    im_color[(~mask).reshape(width,height)] = (0,0,0)
    if mode=='rgb':
        im_color = cv2.cvtColor(im_color.astype(np.float32), cv2.COLOR_BGR2RGB)
    return im_color


def vis_data(imgs, render_dict, img_meta, num_rays, save_dir='./', is_novel=False):
    result = {}
    os.makedirs(os.path.join(save_dir, 'origin_rgb', img_meta['scene_name']), exist_ok=True)
    num_cams, H, W = imgs.size(0), imgs.size(2), imgs.size(3)
    rgbs = []
    for frame in range(num_cams):
        rgb = imgs[frame].permute(1,2,0).cpu().numpy()*255.0
        if 0:
            lidar_depth = np.zeros([H,W])
            mask = img_meta['depth_mask'][frame]
            depth_loc = img_meta['depth_loc'][frame][mask]
            lidar_depth[(depth_loc[:,1]*H).astype(int),(depth_loc[:,0]*W).astype(int)] = img_meta['depth_gt'][frame][mask]
            lidar_depth_rgb = depth2rgbimg(lidar_depth, [H,W])
            rgb[lidar_depth!=0] = lidar_depth_rgb[lidar_depth!=0]
        rgbs.append(rgb)
    save_surroundimg(rgbs, os.path.join(save_dir, 'origin_rgb', img_meta['scene_name'], str(img_meta['index'])+'.jpg'), img_meta['cam_types'], rgb=False)

    if 'sem' in render_dict.keys():
        num_cams = render_dict['sem'][0].size(1)
        sem_folder = os.path.join(save_dir, 'render_semantic', img_meta['scene_name'])
        os.makedirs(sem_folder, exist_ok=True)
        sem_score, semantic = render_dict['sem'][0].max(-1)
        semantic = semantic.cpu().numpy()
        sem_color = semantic_colors[semantic][...,:3].reshape(num_cams, *num_rays, 3)
        save_surroundimg(sem_color, os.path.join(sem_folder, str(img_meta['index'])+'.jpg'), img_meta['cam_types'], rgb=True)

    if 'sem_gt' in render_dict.keys():
        num_cams = render_dict['sem_gt'].size(1)
        semgt_folder = os.path.join(save_dir, 'semantic_label', img_meta['scene_name'])
        os.makedirs(semgt_folder, exist_ok=True)
        semantic_gt = render_dict['sem_gt'].int().cpu().numpy()
        sem_color = semantic_colors[semantic_gt][...,:3].reshape(num_cams, *num_rays, 3)
        save_surroundimg(sem_color, os.path.join(semgt_folder, str(img_meta['index'])+'.jpg'), img_meta['cam_types'], rgb=True)
    
    if 'vis_normal' in render_dict.keys():
        num_cams = render_dict['vis_normal'][0].size(1)
        normal_folder = os.path.join(save_dir, 'render_normal', img_meta['scene_name']) if not is_novel else os.path.join(save_dir, 'render_normal_novel', img_meta['scene_name'])
        os.makedirs(normal_folder, exist_ok=True)
        normal = render_dict['vis_normal'][0].reshape(num_cams, *num_rays, 3).cpu().numpy()
        save_surroundimg(normal*255.0, os.path.join(normal_folder, str(img_meta['index'])+'.jpg'), img_meta['cam_types'], rgb=True)

    if 'ms_depths' in render_dict.keys():
        depth_folder = os.path.join(save_dir, 'render_depth', img_meta['scene_name']) if not is_novel else os.path.join(save_dir, 'render_depth_novel', img_meta['scene_name'])
        os.makedirs(depth_folder, exist_ok=True)
        num_cams = render_dict['ms_depths'][0].size(1)
        depth_rgbs = []
        for frame in range(num_cams):
            depth = render_dict['ms_depths'][0][0,frame].reshape(num_rays).cpu().numpy()
            depth_rgb = depth2rgbimg(depth, num_rays)
            depth_rgbs.append(depth_rgb)
        save_surroundimg(depth_rgbs, os.path.join(depth_folder, str(img_meta['index'])+'.jpg'), img_meta['cam_types'], rgb=False)
    return result
    
def pass_print(*args, **kwargs):
    pass

def main(local_rank, args, frames_info):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg = modify_for_eval(cfg, args.dataset, False, args)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 0:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20513")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, 'eval_depth_' + osp.basename(args.py_config)))
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_depth_{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(args)
    logger.info(f'Config:\n{cfg.pretty_text}')

    import model
    from dataset import get_dataloader

    # build model    
    cfg.model.head.return_max_depth = True
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    # if cfg.get('flow', False):
    #     from utils.unimatch_utils import build_unimatch_model, forward_unimatch_model
    #     unimatch_model = build_unimatch_model(cfg.get('flow_ckpt', 'unimatch/pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth'))
        
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=distributed,
        val_only=True)

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', False)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        logger.info(f'successfully resumed from epoch {ckpt["epoch"]}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        logger.info(f'successfully load from ' + cfg.load_from)
        
    # eval
    print_freq = cfg.print_freq
    my_model.eval()
    if args.depth_metric:
        from utils.metric_util import DepthMetric
        if args.dataset == 'kitti' or args.dataset == 'kitti_raw':
            camera_names = ['front']
        elif args.dataset == 'nuscenes':
            camera_names = ['front', 'front_right', 'front_left', \
                            'back',  'back_left',   'back_right']
        depth_metric = DepthMetric(
            camera_names=camera_names).cuda()
        depth_metric._reset()
        
    if args.vis_occupancy:
        occ_save_path = os.path.join(args.work_dir, "occupancy")
        os.makedirs(occ_save_path, exist_ok=True)
    if args.vis_render:
        render_depth_path = os.path.join(args.work_dir, "render_depth")
        os.makedirs(render_depth_path, exist_ok=True)
        render_normal_path = os.path.join(args.work_dir, "render_normal")
        os.makedirs(render_normal_path, exist_ok=True)
        origin_rgb_path = os.path.join(args.work_dir, "origin_rgb")
        os.makedirs(origin_rgb_path, exist_ok=True)

    with torch.no_grad():
        for i_iter_val, (input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, \
                         img_metas, curr_aug, prev_aug, next_aug, lidar_rays) in enumerate(val_dataset_loader):
            input_imgs = input_imgs.cuda()
            with torch.cuda.amp.autocast(amp):
                results = my_model(imgs=input_imgs, metas=img_metas, prepare=True)
                if args.novel_view:
                    import math
                    beta = -10 /180*math.pi
                    rot = np.array([[math.cos(beta),0,math.sin(beta)],[0,1,0],[-math.sin(beta),0,math.cos(beta)]])
                    T = np.eye(4)
                    T[:3,:3] = rot
                    T[:3,3] = np.array([0,0,2.0])
                    img_metas[0].update({'temImg2lidar': img_metas[0]['cam2lidar'] @ T[None,...] @ np.linalg.inv(img_metas[0]['intrinsic']) @ np.linalg.inv(img_metas[0]['render_img_aug_matrix'])})
                
                if args.depth_metric or args.vis_render:
                    result_dict = raw_model.head.render(**results, batch=args.batch)
                if args.vis_render:
                    result_dict.update(results)
                    result = vis_data(curr_imgs[0], result_dict, img_metas[0], cfg.num_rays, save_dir=args.work_dir, is_novel=args.novel_view)
                if args.depth_metric:
                    #### calculate all sorts of depths
                    ms_depths = result_dict['ms_depths'][0]
                    ms_depths = ms_depths.unflatten(-1, cfg.num_rays)

                    if 'ms_depths_median' in result_dict:
                        ms_depths_median = result_dict['ms_depths_median'][0]
                        ms_depths_median = ms_depths_median.unflatten(-1, cfg.num_rays)
                    if 'ms_max_depths' in result_dict:
                        ms_depths_max = result_dict['ms_max_depths'][0]
                        ms_depths_max = ms_depths_max.unflatten(-1, cfg.num_rays)
                        
                    depth_loc = ms_depths.new_tensor(img_metas[0]['depth_loc'])
                    depth_gt = ms_depths.new_tensor(img_metas[0]['depth_gt'])
                    depth_mask = torch.from_numpy(img_metas[0]['depth_mask']).cuda()
                    if args.depth_metric_tgt == 'raw':
                        depth_pred = ms_depths[0]
                    elif args.depth_metric_tgt == 'median':
                        depth_pred = ms_depths_median[0]
                    elif args.depth_metric_tgt == 'max':
                        depth_pred = ms_depths_max[0]
                    depth_metric._after_step(depth_loc, depth_gt, depth_mask, depth_pred)

                if args.vis_occupancy:
                    result_dict = my_model(imgs=input_imgs, metas=img_metas, aabb=cfg['point_cloud_range'], resolution=args.resolution, occ_only=True)
                    pred_occ = (result_dict['sdf'] <= args.thresh).to(torch.int) if 'sdf' in result_dict else (result_dict['occ'] >= 0.3).to(torch.int)
                    if args.dataset == 'kitti':
                        pred_occ[..., 28:] = 0
                        pred_occ[-6:, ...] = 0
                        pred_occ[:, :6, :] = 0
                        pred_occ[:, -6:, :] = 0
                        os.makedirs(os.path.join(occ_save_path, img_metas[0]['scene_name']), exist_ok=True)
                        pred_path = os.path.join(occ_save_path, img_metas[0]['scene_name'], img_metas[0]['token']+'.npz')
                        save_dict = {'occ': pred_occ.cpu().numpy()}
                        np.savez_compressed(pred_path, **save_dict)
                    elif args.dataset == 'nuscenes':
                        pred_occ[..., 16:] = 0
                        pred_occ[:6, ...] = 0
                        pred_occ[-6:, ...] = 0
                        pred_occ[:, :6, :] = 0
                        pred_occ[:, -6:, :] = 0
                        os.makedirs(os.path.join(occ_save_path, img_metas[0]['scene_name']), exist_ok=True)
                        pred_path = os.path.join(occ_save_path, img_metas[0]['scene_name'], img_metas[0]['token']+'.npz') # +'_'+str(img_metas[0]['index'])
                        save_dict = {'occ': pred_occ.cpu().numpy()}
                        if 'sem' in result_dict.keys():
                            sem = result_dict['sem']
                            sem[pred_occ==0] = 17
                            save_dict.update(sem=sem.cpu().numpy().astype(int))
                        if 'flow' in result_dict.keys():
                            save_dict.update(flow=result_dict['flow'].cpu().numpy())        
                        np.savez_compressed(pred_path, **save_dict)
                        frames_info.append({
                            'pred_path': pred_path,
                            'scene_name': img_metas[0]['scene_name'],
                            'token': img_metas[0]['token'],
                            'keyframe_index': img_metas[0]['keyframe_index']
                        })
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d/%d'%(i_iter_val, len(val_dataset_loader)))
    
    if args.depth_metric:
        depth_metric._after_epoch()

def save_pkl(list_m, save_path):
    print('saving frames pkl......')
    infos = {}
    infos['infos'] = list(list_m)
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)

if __name__ == '__main__':
    # Training settings
    m = torch.multiprocessing.Manager()
    frames_info = m.list()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/kitti/kitti_occ_odom.py')
    parser.add_argument('--work-dir', type=str, default='out/visualization/kitti/occ_odom')
    parser.add_argument('--resume-from', type=str, default='') 
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occupancy', default=1)
    parser.add_argument('--vis-render', default=0)
    parser.add_argument('--novel-view', default=0)
    parser.add_argument('--resolution', type=float, default=0.2) # kitti: 0.2, nuscenes: 0.4
    parser.add_argument('--thresh', type=float, default=0.1)
    parser.add_argument('--depth-metric', default=0) # , action='store_true'
    parser.add_argument('--depth-metric-tgt', type=str, default='raw')
    # parser.add_argument('--dataset', type=str, default='nuscenes') # kitti, nuscenes
    parser.add_argument('--dataset', type=str, default='kitti') 
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--num-rays', type=int, nargs='+', default=[128, 352])
    # parser.add_argument('--num-rays', type=int, nargs='+', default=[480,640]) # [176, 608]
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.hfai:
        os.environ['HFAI'] = 'true'
    try:
        if ngpus > 1:
            torch.multiprocessing.spawn(main, args=(args, frames_info,), nprocs=args.gpus)
        else:
            main(0, args, frames_info)
        save_pkl(frames_info, os.path.join(args.work_dir, "frames_info.pkl"))
    except KeyboardInterrupt as e:
        save_pkl(frames_info, os.path.join(args.work_dir, "frames_info.pkl"))

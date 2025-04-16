import time, argparse, os.path as osp
import os, math, pickle
os.environ["PYTHONWARNINGS"] ='ignore:semaphore_tracker:UserWarning'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch, numpy as np
import torch.distributed as dist
import cv2
import open3d as o3d
import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from utils.feat_tools import multi2single_scale
from copy import deepcopy
from utils.config_tools import modify_for_eval
from utils.flow_viz import save_vis_flow
# select_frames = []
from cotracker.utils.visualizer import Visualizer

def depth2rgbimg(depth_img, img_size, max_depth=None, mode='bgr'):
    import cv2
    width, height = img_size[0], img_size[1]
    mask = np.logical_and(depth_img>0, depth_img<80)
    disp_img = np.zeros_like(depth_img)
    disp_img[mask] = 1.0 / depth_img[mask]
    if mask.sum()>0:
        # disp_min = disp_img[mask].min()
        disp_min = 1.0 / 120.0
        disp_max = np.percentile(disp_img[mask], 95)
        disp_img[mask] = 255.0*(disp_img[mask]-disp_min)/(disp_max-disp_min)
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_img.reshape(width,height,1), alpha=1), cv2.COLORMAP_MAGMA)
    im_color[(~mask).reshape(width,height)] = (0,0,0)
    if mode=='rgb':
        im_color = cv2.cvtColor(im_color.astype(np.float32), cv2.COLOR_BGR2RGB)
    return im_color

def cal_pixel(trans, coords, img_size):
    trans = trans.float()
    coords = coords.float()
    eps = 1e-5
    pixel = torch.matmul(trans, coords).squeeze(-1) # bs, N, R, 4
    mask = pixel[..., 2] > 0
    pixel = pixel[..., :2] / torch.maximum(torch.ones_like(pixel[..., :1]) * eps, pixel[..., 2:3])
    mask = mask & (pixel[..., 0] > 0) & (pixel[..., 0] <img_size[1]) & \
                    (pixel[..., 1] > 0) & (pixel[..., 1] < img_size[0])
    return pixel, mask

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args, select_frames):
    # global settings
    time.sleep(1)
    # global select_frames
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
        port = os.environ.get("MASTER_PORT", "2043")
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
        cfg.dump(osp.join(args.work_dir, 'precompute_tracking' + osp.basename(args.py_config)))
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'eval_depth_{timestamp}.log')
    logger = MMLogger('occflow', log_file=log_file)
    MMLogger._instance_dict['occflow'] = logger
    logger.info(args)
    logger.info(f'Config:\n{cfg.pretty_text}')
    tracking_flow_path = cfg.get('tracking_flow_path', '/mnt/meadow/lyl/nuscenes_flow3')
    for cam in cfg.cam_types:
        os.makedirs(os.path.join(tracking_flow_path, f'samples/{cam}'), exist_ok=True)

    from dataset import get_dataloader

    if cfg.get('flow', False):
        from utils.unimatch_utils import build_unimatch_model, forward_unimatch_model
        unimatch_model = build_unimatch_model(cfg.get('flow_ckpt', '/sdb1/lyl/unimatch/pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth'))
    if cfg.get('tracking', False):
        # from cotracker.predictor import CoTrackerPredictor
        # if args.use_v2_model:
        #     window_len = 8
        # elif args.offline:
        #     window_len = 60
        # else:
        #     window_len = 16
        # cotracker = CoTrackerPredictor(checkpoint=cfg.get('cotracker_ckpt', '/home/lyl/myprojects/co-tracker/checkpoints/cotracker2.pth'),
        #                                v2=args.use_v2_model,
        #                                offline=args.offline,
        #                                window_len=window_len).cuda()
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2v1").cuda()
        
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
    logger.info('work dir: ' + args.work_dir)
        
    # eval
    print_freq = cfg.print_freq
    
    with torch.no_grad():
        for i_iter_val, (curr_imgs, next_imgs, depth_loc, depth_gt, depth_mask, img_metas) in tqdm(enumerate(val_dataset_loader)):
            if i_iter_val % 100 == 0:
                torch.cuda.empty_cache()
            time.sleep(0.1)
            folder = img_metas[0]['scene_token']+'_'+str(img_metas[0]['index'])
            scene_token = img_metas[0]['scene_token']
            temporal_index = str(img_metas[0]['index'])
            img_paths = ''
            for i in range(len(cfg.cam_types)):
                img_paths += img_metas[0]['curr_imgs_path'][i].split('/')[-1][:-4] + ' '
            select_frames.append(f'{scene_token} {temporal_index} {img_paths}')
            # if os.path.exists(os.path.join(tracking_flow_path, img_metas[0]['curr_imgs_path'][-1][:-4]+'_flow.npz')):
            #     logger.info(f'[EVAL] {folder} exist!!!')
            #     continue
            logger.info(f'[EVAL] {folder}')
            if cfg.get('flow', False):
                B, N, C, H, W = curr_imgs.size()
                next_imgs = next_imgs.cuda().reshape(B,-1,N,C,H,W)[:,-4,...]
                curr_imgs = curr_imgs.cuda()
                flow_results = forward_unimatch_model(unimatch_model, curr_imgs[0], None, next_imgs[0])
                flow_curr_next = flow_results['flow_curr_next']
                for frame in range(flow_curr_next.size(1)):
                    save_vis_flow(flow_curr_next[0,frame], output_path=os.path.join('./saved_videos', f'flow_{frame}.png'))
            if cfg.get('tracking', False):
                next_imgs = next_imgs.cuda()
                curr_imgs = curr_imgs.cuda()
                B, N, C, H, W = curr_imgs.size()
                videos = torch.cat((curr_imgs, next_imgs), dim=1).reshape(B,-1,N,C,H,W).permute(0,2,1,3,4,5).flatten(0,1).flip(dims=[2])
                if 'sem' in img_metas[0] or 'ins' in img_metas[0]:
                    semantic = torch.stack([torch.from_numpy(np.asarray(img_metas[i]['ins'])) for i in range(len(img_metas))]).cuda().reshape(-1,1,H,W)
                    # dynamic_mask = (semantic==3) | (semantic==4) | (semantic==10) | (semantic==2) | (semantic==7) | (semantic==6)
                    # dynamic_mask = semantic != 0
                    stride = 10
                    grid_y, grid_x = torch.meshgrid(torch.linspace(0,H-1,H//stride), torch.linspace(0,W-1,W//stride))
                    grid = torch.stack([grid_x, grid_y], dim=-1).cuda()
                    grid_norm = grid.clone()
                    grid_norm[...,0] = 2*grid_norm[...,0]/(W-1)-1.0
                    grid_norm[...,1] = 2*grid_norm[...,1]/(H-1)-1.0
                    sample_mask = torch.nn.functional.grid_sample(semantic.float(), grid_norm[None].repeat(N,1,1,1),
                                                                  align_corners=True, mode='nearest', padding_mode='border') # .bool()
                del curr_imgs, next_imgs
                for frame in range(videos.size(0)): # [4]
                    video = videos[frame:frame+1].detach()
                    sample = grid[sample_mask[frame,0]>0]
                    ins_mask = sample_mask[frame,0][sample_mask[frame,0]>0]
                    sample = torch.cat((torch.zeros_like(sample[:,0:1]), sample), dim=1).unsqueeze(0)
                    pred_tracks, pred_visibility = cotracker(video, queries=sample, grid_query_frame=0)
                    # backward

                    last = pred_tracks.size(1)-2 if not img_metas[0]['scene_final'] else pred_tracks.size(1)
                    num = len(list(range(1, last, 1)))
                    flow = 1e7 * torch.ones([num, H//stride, W//stride, 2]).cuda()
                    pred_flow = (pred_tracks[0, 1:last:1] - pred_tracks[0, 0:1])/2
                    flow[(sample_mask[frame,0]>0)[None,...].repeat(num,1,1)] = pred_flow.reshape(-1,2)
                    filename = os.path.join(tracking_flow_path, img_metas[0]['curr_imgs_path'][frame][:-4]+'_flow.npz')
                    np.savez_compressed(filename, flow=flow.cpu().numpy().astype(np.float16))

                    if args.visualize and pred_tracks.size(-2)>0 and i_iter_val % 5 == 0:
                        save_dir = os.path.join(args.work_dir, img_metas[0]['scene_name'], str(img_metas[0]['index']))
                        os.makedirs(save_dir, exist_ok=True)
                        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=1, fps=1, mode='cool', show_first_frame=3)
                        vis.visualize(video[:,:last], pred_tracks[:,:last], pred_visibility[:,:last], query_frame=0, filename=f'video_{frame}')
                # torch.cuda.empty_cache()
                    
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d/%d'%(i_iter_val, len(val_dataset_loader)))

def save_select(list_m, save_path):
    print('saving select frames......')
    with open(save_path, 'w') as f:
        for i in list_m:
            f.write(i+'\n')

if __name__ == '__main__':
    # Training settings
    m = torch.multiprocessing.Manager()
    select_frames = m.list()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/nuscenes/nuscenes_tracking.py')
    parser.add_argument('--work-dir', type=str, default='out/visualization/nuscenes/nuscenes_tracking')
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize', default=1)
    parser.add_argument('--use-v2-model', default=1)
    parser.add_argument('--offline', default=0)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    # parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--num-rays', type=int, nargs='+', default=[128, 352]) # [225,400]
    # parser.add_argument('--num-rays', type=int, nargs='+', default=[88, 304]) # [225,400]
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.hfai:
        os.environ['HFAI'] = 'true'
    try:
        if ngpus > 1:
            torch.multiprocessing.spawn(main, args=(args, select_frames,), nprocs=args.gpus)
        else:
            main(0, args, select_frames)
    except KeyboardInterrupt as e:
        save_select(select_frames, os.path.join(args.work_dir, "select_frames.txt"))
    
    

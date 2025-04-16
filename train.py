# import multiprocessing
# multiprocessing.set_start_method('spawn')
import time, argparse, os.path as osp, os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["NCCL_P2P_DISABLE"] = "1"
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy
import shutil
import mmcv
# mmcv.use_backend('pillow')
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

import warnings
warnings.filterwarnings("ignore")

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 0:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20509")
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
        world_size = 1
    
    if local_rank == 0:
        # from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        # writer = SummaryWriter(log_dir=osp.join(args.work_dir, 'tf'))
        from utils.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('occflow', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['occflow'] = writer
        if os.path.exists(osp.join(args.work_dir, 'code')):
            shutil.rmtree(osp.join(args.work_dir, 'code'))
        os.makedirs(osp.join(args.work_dir, 'code'))
        shutil.copytree('dataset', osp.join(args.work_dir, 'code' + '/dataset'))
        shutil.copytree('loss', osp.join(args.work_dir, 'code' + '/loss'))
        shutil.copytree('model', osp.join(args.work_dir, 'code' + '/model'))
        shutil.copytree('utils', osp.join(args.work_dir, 'code' + '/utils'), ignore=shutil.ignore_patterns('*pyc*', 'build'))
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('occflow', log_file=log_file)
    MMLogger._instance_dict['occflow'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS
    from utils.feat_tools import multi2single_scale

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

    # if args.dataset == 'nuscenes' and cfg.get('sem', False):
    #     from utils.openseed_utils import build_openseed_model, forward_openseed_model
    #     openseed_model = build_openseed_model()
    if cfg.get('flow', False) and not cfg.get('tracking', False):
        from utils.unimatch_utils import build_unimatch_model, forward_unimatch_model
        unimatch_model = build_unimatch_model(cfg.get('flow_ckpt', 'ckpts/gmflow-scale2-regrefine6-kitti15-25b554d7.pth'))

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        cfg.nusc,
        dist=distributed,
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    # cfg.loss.update({'writer': writer})
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config
        )
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)
    amp = cfg.get('amp', False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
        os.environ['amp'] = 'true'
    else:
        os.environ['amp'] = 'false'
    
    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

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
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        loss_func.iter_counter = global_iter
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        logger.info(f'successfully load from ' + cfg.load_from)
        
    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_accumulation = args.gradient_accumulation
    # assert grad_accumulation * world_size == 8
    grad_norm = 0
    if args.depth_metric:
        from utils.metric_util import DepthMetric
        if args.dataset == 'kitti':
            camera_names = ['front']
        elif args.dataset == 'nuscenes':
            camera_names = ['front', 'front_right', 'front_left', \
                            'back',  'back_left',   'back_right']
        depth_metric = DepthMetric(
            camera_names=camera_names).cuda()
        depth_metric._reset()

    while epoch < max_num_epochs:
        my_model.train()
        os.environ['eval'] = 'false'
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, (
            input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, 
            img_metas, curr_aug, prev_aug, next_aug, lidar_rays) in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter
            
            input_imgs = input_imgs.cuda()
            curr_imgs = curr_imgs.cuda()
            prev_imgs = prev_imgs.cuda()
            next_imgs = next_imgs.cuda()
            color_imgs = color_imgs.cuda()
            if lidar_rays != None:
                lidar_rays = lidar_rays.cuda()
                lidar_ranges = lidar_rays[...,-1]
                lidar_rays = lidar_rays[...,:-1]
            data_time_e = time.time()

            with torch.cuda.amp.autocast(amp):
                # forward + backward + optimize                
                # if args.dataset == 'nuscenes' and cfg.get('sem', False):
                #     sem_map = forward_openseed_model(openseed_model, curr_imgs[0] * 256., cfg.img_size)
                #     img_metas[0].update({'sem': sem_map})
                if cfg.get('flow', False) and epoch>=cfg.get('flow_epoch',0):
                    if not cfg.get('tracking', False):
                        prev_input = prev_imgs[0]*256. if cfg.get('with_prev_flow', False) else None
                        next_input = next_imgs[0]*256. if not cfg.get('tracking', False) else None
                        flow_results = forward_unimatch_model(unimatch_model, curr_imgs[0]*256., prev_input, next_input)
                    else:
                        flow_results = {}
                        tracking_flow = torch.stack([torch.from_numpy(np.asarray(img_metas[i]['tracking_flow'])) for i in range(len(img_metas))]).float().cuda().permute(0,1,4,2,3)
                        flow_results.update(flow_curr_next=tracking_flow)
                curr_feats, prev_feats, next_feats = curr_imgs, prev_imgs, next_imgs

                result_dict = my_model(
                    imgs=input_imgs, metas=img_metas, global_iter=global_iter, lidar_rays=lidar_rays)

                loss_input = {
                    'curr_imgs': curr_imgs,
                    'prev_imgs': prev_imgs,
                    'next_imgs': next_imgs,
                    'curr_feats': curr_feats,
                    'prev_feats': prev_feats,
                    'next_feats': next_feats,
                    'metas': img_metas,
                    'color_imgs': color_imgs,
                }
                try:
                    loss_input.update(model=my_model.module.head.model)
                except:
                    pass
                if cfg.get('flow', False) and epoch>=cfg.get('flow_epoch',0):
                    # loss_input.update(flow_curr_prev=flow_curr_prev, flow_curr_next=flow_curr_next)
                    loss_input.update(flow_results)
                if lidar_rays != None:
                    loss_input.update(lidar_ranges=lidar_ranges)
                if 'input_depth_imgs' in img_metas[0]:
                    input_depth_imgs = torch.stack([torch.from_numpy(img_metas[i]['input_depth_imgs']) for i in range(len(img_metas))]).cuda()
                    loss_input.update(input_depth_imgs=input_depth_imgs)
                if 'flow_3d' in img_metas[0]:
                    flow_3d_gt = torch.stack([torch.from_numpy(img_metas[i]['flow_3d']) for i in range(len(img_metas))]).cuda()
                    occ_mask = torch.stack([torch.from_numpy(img_metas[i]['occ_mask']) for i in range(len(img_metas))]).cuda()
                    loss_input.update(flow_3d_gt=flow_3d_gt, occ_mask=occ_mask)
                if 'sem' in img_metas[0]:
                    semantic_mask = torch.stack([torch.from_numpy(np.asarray(img_metas[i]['sem'])) for i in range(len(img_metas))]).cuda()
                    loss_input.update(semantic_mask=semantic_mask)
                if 'ins' in img_metas[0]:
                    instance_mask = torch.stack([torch.from_numpy(np.asarray(img_metas[i]['ins'])) for i in range(len(img_metas))]).cuda()
                    loss_input.update(instance_gt=instance_mask)
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    loss_input.update({
                        loss_input_key: result_dict[loss_input_val]})
                loss, loss_dict = loss_func(loss_input)

                loss = loss / grad_accumulation
            if not amp:
                loss.backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            # print('[TRAIN] Epoch %d Iter %5d/%d: rank_id: %d time: %.3f (%.3f)'%(
            #         epoch, i_iter, len(train_dataset_loader), local_rank, time_e - time_s, data_time_e - data_time_s))
            if i_iter % print_freq == 0 and local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                detailed_loss = []
                for loss_name, loss_value in loss_func.loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {(sum(loss_value)/len(loss_value)):.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []
                loss_func.clear_loss_dict()
            data_time_s = time.time()
            time_s = time.time()

            if args.iter_resume:
                if (i_iter + 1) % 50 == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)
                    logger.info(f'iter ckpt {i_iter + 1} saved!')
        
        epoch += 1
        first_run = False
        
        # save checkpoint
        if local_rank == 0 and (epoch % cfg.get('save_every_epochs', 1) == 0 or epoch==max_num_epochs):
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)
        
        if local_rank == 0 and epoch==max_num_epochs:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'epoch': epoch,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'state_{epoch}.pth')
            torch.save(dict_to_save, save_file_name)
        
        # eval
        if epoch % cfg.get('eval_every_epochs', 1) != 0:
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []

        with torch.no_grad():
            for i_iter_val, (input_imgs, curr_imgs, prev_imgs, next_imgs, color_imgs, \
                             img_metas, curr_aug, prev_aug, next_aug, lidar_rays) in enumerate(val_dataset_loader):
                
                input_imgs = input_imgs.cuda()
                curr_imgs = curr_imgs.cuda()
                prev_imgs = prev_imgs.cuda()
                next_imgs = next_imgs.cuda()
                color_imgs = color_imgs.cuda()
                
                with torch.cuda.amp.autocast(amp):

                    # if args.dataset == 'nuscenes' and cfg.get('sem', False):
                    #     sem_map = forward_openseed_model(openseed_model, curr_imgs[0] * 256., cfg.img_size)
                    #     img_metas[0].update({'sem': sem_map})

                    curr_feats, prev_feats, next_feats = curr_imgs, prev_imgs, next_imgs

                    result_dict = my_model(imgs=input_imgs, metas=img_metas, lidar_rays=None)

                    loss_input = {
                        'curr_imgs': curr_imgs,
                        'prev_imgs': prev_imgs,
                        'next_imgs': next_imgs,
                        'curr_feats': curr_feats,
                        'prev_feats': prev_feats,
                        'next_feats': next_feats,
                        'metas': img_metas,
                        'color_imgs': color_imgs,
                    }
                    # for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    #     loss_input.update({
                    #         loss_input_key: result_dict[loss_input_val]
                    #     })
                    # loss, loss_dict = loss_func(loss_input)

                if args.depth_metric:
                    ms_depths = result_dict['ms_depths'][0]
                    ms_depths = ms_depths.unflatten(-1, cfg.num_rays)

                    depth_loc = ms_depths.new_tensor(img_metas[0]['depth_loc'])
                    depth_gt = ms_depths.new_tensor(img_metas[0]['depth_gt'])
                    depth_mask = torch.from_numpy(img_metas[0]['depth_mask']).cuda()
                    # depth_pred = ms_depths[0, :(ms_depths.shape[1] // 2)]
                    depth_pred = ms_depths[0]
                    depth_metric._after_step(depth_loc, depth_gt, depth_mask, depth_pred)
                
                # val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d'%(epoch, i_iter_val))
                    # logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                    #     epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    # detailed_loss = []
                    # for loss_name, loss_value in loss_dict.items():
                    #     detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    # detailed_loss = ', '.join(detailed_loss)
                    # logger.info(detailed_loss)
        if args.depth_metric:
            depth_metric._after_epoch()
            depth_metric._reset()
        # logger.info('Current val loss is %.3f' %
        #         (np.mean(val_loss_list)))
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/nuscenes/nuscenes_occ_flow_voxelaffm.py')
    parser.add_argument('--work-dir', type=str, default='out/train/nuscenes/occ_flow')
    parser.add_argument('--resume-from', type=str, default='') 
    parser.add_argument('--hfai', action='store_true', default=False)
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--depth-metric', default=True) # action='store_true', 
    parser.add_argument('--dataset', type=str, default='nuscenes')
    # parser.add_argument('--dataset', type=str, default='kitti')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.hfai:
        os.environ['HFAI'] = 'true'

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)

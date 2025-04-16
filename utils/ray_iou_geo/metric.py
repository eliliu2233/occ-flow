import numpy as np
import os
from tqdm import tqdm
import pickle, gzip
import argparse
import csv
occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
    ]

def calc_metrics(pred_cls_list, pred_dist_list, pred_flow_list, pred_points_list, gt_cls_list, gt_dist_list, gt_flow_list, gt_points_list):
    occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
    ]

    flow_class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian',
    ]
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

    ave = np.zeros([len(thresholds), len(occ_class_names)])
    for i, cls in enumerate(occ_class_names):
        if cls not in flow_class_names:
            ave[:, i] = np.nan

    ave_count = np.zeros([len(thresholds), len(occ_class_names)])

    for idx in tqdm(range(len(pred_cls_list))):
        for j, threshold in enumerate(thresholds):
            pred_cls = pred_cls_list[idx].astype(np.int32)
            pred_dist = pred_dist_list[idx].astype(np.float32)
            pred_flow = pred_flow_list[idx].astype(np.float32)
            
            gt_cls = gt_cls_list[idx].astype(np.int32)
            gt_dist = gt_dist_list[idx].astype(np.float32)
            gt_flow = gt_flow_list[idx].astype(np.float32)

            valid_mask = (gt_cls != len(occ_class_names) - 1)
            pred_cls = pred_cls[valid_mask]
            pred_dist = pred_dist[valid_mask]
            pred_flow = pred_flow[valid_mask]

            gt_cls = gt_cls[valid_mask]
            gt_dist = gt_dist[valid_mask]
            gt_flow = gt_flow[valid_mask]
            
            if len(pred_points_list)>0 and len(gt_points_list)>0:
                pred_points = pred_points_list[idx].astype(np.float32)
                pred_points = pred_points[valid_mask]
                gt_points = gt_points_list[idx].astype(np.float32)
                gt_points = gt_points[valid_mask]

            # L1
            l1_error = np.abs(pred_dist - gt_dist)
            tp_dist_mask = (l1_error < threshold)
            
            for i, cls in enumerate(occ_class_names):
                cls_id = occ_class_names.index(cls)
                # without semantic evaluation
                # cls_mask_pred = (pred_cls == cls_id)
                cls_mask_pred = (gt_cls == cls_id)
                cls_mask_gt = (gt_cls == cls_id)

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()

                # flow L2 error
                if cls in flow_class_names and tp_mask.sum() > 0:
                    flow_error = np.linalg.norm(gt_flow[tp_mask] - pred_flow[tp_mask], axis=1)
                    ave[j][i] += np.sum(flow_error)
                    ave_count[j][i] += flow_error.shape[0]
    
    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    ave_list = [ave[1][:-1] / ave_count[1][:-1], ave[2][:-1] / ave_count[2][:-1]]  # use threshold = 2

    return iou_list, ave_list

def compute(args):
    print("Evaluating...")

    with gzip.open(os.path.join(args.work_dir, args.pred), 'rb') as f:  
        pred_file = pickle.load(f)

    with gzip.open(os.path.join(args.work_dir, args.gt), 'rb') as f:
        openocc_test_file = pickle.load(f)
        
    # with gzip.open('/mnt/meadow/lyl/ray_iou_output/occnerf_step2_f5/nuscenes_infos_val_occ_pcd.gz', 'rb') as f:
    #     ref_file = pickle.load(f)

    print("Start to evaluate on nuScenes OpenOcc...")
    pred_cls_list = []
    pred_dist_list = []
    pred_flow_list = []
    pred_points_list = []
    gt_cls_list = []
    gt_dist_list = []
    gt_flow_list = []
    gt_points_list = []
    count = 0
    for gt_token in tqdm(openocc_test_file['results'].keys()):
        # if gt_token not in ref_file['results'].keys():
        #     continue

        if gt_token in pred_file['results'].keys():  # found
            pred_data = pred_file['results'][gt_token]
            gt_data = openocc_test_file['results'][gt_token]

            pred_cls_list.append(pred_data['pcd_cls'])
            pred_dist_list.append(pred_data['pcd_dist'])
            pred_flow_list.append(pred_data['pcd_flow'])
            if 'pcd_points' in pred_data.keys():
                pred_points_list.append(pred_data['pcd_points'])

            gt_cls_list.append(gt_data['pcd_cls'])
            gt_dist_list.append(gt_data['pcd_dist'])
            gt_flow_list.append(gt_data['pcd_flow'])
            if 'pcd_points' in gt_data.keys():
                gt_points_list.append(gt_data['pcd_points'])
            count += 1
            # print(gt_token)
            # if count>118:
            #     break
        else:
            raise RuntimeError(f'OpenOcc: prediction is not found for token: {gt_token}')

    openocc_iou_list, openocc_ave_list = calc_metrics(pred_cls_list, pred_dist_list, pred_flow_list, pred_points_list, gt_cls_list, gt_dist_list, gt_flow_list, gt_points_list)

    openocc_miou = np.nanmean(openocc_iou_list)
    openocc_mave = np.nanmean(openocc_ave_list[0])
    openocc_mave2 = np.nanmean(openocc_ave_list[1])
    openocc_occ_score = openocc_miou * 0.9 + max(1 - openocc_mave, 0.0) * 0.1

    # final score
    occ_score = openocc_occ_score

    output = {
        "RayIoU@1": np.nanmean(openocc_iou_list[0]),
        "RayIoU@2": np.nanmean(openocc_iou_list[1]),
        "RayIoU@4": np.nanmean(openocc_iou_list[2]),
        "RayIoU": openocc_miou,
        "mAVE@2": openocc_mave,
        "mAVE@4": openocc_mave2,
        "final_Occ_Score": occ_score
    }

    evaluation = {
        "public_score": output,
        "private_score": output
    }

    print(output)
    
    with open(os.path.join(args.work_dir, "all_results.csv"), "wt") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['classes']+occ_class_names)
        writer.writerow(['rayiou@1']+list(openocc_iou_list[0]))
        writer.writerow(['rayiou@2']+list(openocc_iou_list[1]))
        writer.writerow(['rayiou@4']+list(openocc_iou_list[2]))
        writer.writerow(['ave']+list(openocc_ave_list))
    
    with open(os.path.join(args.work_dir, "results.txt"), 'w') as f:
        for k,v in output.items():
            f.write(f'{k}: {v}\n')

    print('End of evaluation.')
    return evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default='ray_iou_output/occ_flow')
    parser.add_argument("--pred", default='my_pred_pcd.gz')
    parser.add_argument("--gt", default='nuscenes_infos_val_occ_pcd.gz')
    args = parser.parse_args()

    compute(args)

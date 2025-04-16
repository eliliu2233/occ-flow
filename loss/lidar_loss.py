from .base_loss import BaseLoss
import torch
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class LidarLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'lidar_ranges': 'lidar_ranges',
                'pred_ranges': 'pred_ranges',
            }
        else:
            self.input_dict = input_dict
        self.discard_outliers = kwargs.get('discard_outliers', 0)
        self.discard_outliers_median = kwargs.get('discard_outliers_median', 0)
        self.silog = kwargs.get('silog', False)
        self.loss_func = self.lidar_loss
    
    def lidar_loss(self, lidar_ranges, pred_ranges):
        bs, ray_nums = lidar_ranges.size()
        range_mask = torch.ones_like(lidar_ranges, dtype=bool, device=lidar_ranges.device)
        if self.discard_outliers > 0:
            # Optionally discard a percentage of beam with largest depth errors
            with torch.no_grad():
                depth_err_l1 = torch.nn.functional.l1_loss(pred_ranges, lidar_ranges, reduction='none')
                _, sort_inds = torch.sort(depth_err_l1.data, dim=1) # depth_err[sort_inds]: From small to large           
                dicard_ray_inds = sort_inds[:,-int(ray_nums * self.discard_outliers):]
                range_mask[torch.arange(bs)[:,None], dicard_ray_inds] = False
        if self.discard_outliers_median > 0:
            # Optionally discard beams that have errors exceeding `discard_outliers_median` times the median error.
            assert bs == 1
            with torch.no_grad():
                depth_err_l1 = torch.nn.functional.l1_loss(pred_ranges, lidar_ranges, reduction='none')
                sort_values, sort_inds = torch.sort(depth_err_l1.data, dim=1) # depth_err[sort_inds]: From small to large
                median = sort_values[:, ray_nums//2]
                range_mask[torch.arange(bs)[:,None], depth_err_l1 > median * self.discard_outliers_median] = False
        pred_ranges = pred_ranges[range_mask]
        lidar_ranges = lidar_ranges[range_mask]
        if self.silog:
            # d = torch.abs(torch.log(pred_ranges) - torch.log(lidar_ranges))
            # lidar_loss = torch.sqrt((d ** 2).mean() + 0.85 * (d.mean() ** 2))
            d = torch.log(pred_ranges) - torch.log(lidar_ranges)
            lidar_loss = torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2))
        else:
            lidar_loss = torch.nn.functional.l1_loss(pred_ranges, lidar_ranges, reduction='none').mean()
            # lidar_loss = torch.log(torch.nn.functional.l1_loss(pred_ranges, lidar_ranges, reduction='none')+1).mean()
        if torch.isnan(lidar_loss).sum()>0:
            debug = 1
        return lidar_loss
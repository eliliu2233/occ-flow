from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')

from .multi_loss import MultiLoss
from .rgb_loss_ms import RGBLossMS, SemLossMS, SemCELossMS, SemLoss
from .reproj_loss_mono_multi_new import ReprojLossMonoMultiNew
from .reproj_loss_mono_multi_new_combine import ReprojLossMonoMultiNewCombine, ReprojLossMono
from .reproj_loss_flow import ReprojLossFlow
from .flow_loss import FlowLoss
from .occ_loss_3d import OccLoss3d
from .flow_reg_loss_3d import FlowRegLoss3d
from .flow_reg_loss_2d import FlowRegLoss2d
from .edge_loss_3d_ms import EdgeLoss3DMS, EdgeLossFlow
from .eikonal_loss import EikonalLoss
from .lidar_loss import LidarLoss
from .sparsity_loss import SparsityLoss, HardSparsityLoss, SoftSparsityLoss, AdaptiveSparsityLoss, OccEntropyLoss
from .second_grad_loss import SecondGradLoss

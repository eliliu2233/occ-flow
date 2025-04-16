import torch.nn as nn
from . import OPENOCC_LOSS
from utils.tb_wrapper import WrappedTBWriter
if 'occflow' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('occflow')
else:
    writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs, count_freq=10):
        super().__init__()
        
        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)
        
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0
        self.count_freq = count_freq
        self.loss_dict = {loss_func.__class__.__name__:[] for loss_func in self.losses}
        self.loss_dict.update(total_loss=[])

    def forward(self, inputs):
        
        loss_dict = {}
        tot_loss = 0.
        inputs.update(iters=self.iter_counter)
        for loss_func in self.losses:
            loss_name = loss_func.__class__.__name__
            ret = loss_func(inputs)
            if isinstance(ret, tuple):
                inputs.update(ret[1])
                loss = ret[0]
            else:
                loss = ret
            tot_loss += loss
            # loss_dict.update({
            #     loss_func.__class__.__name__: \
            #     loss.detach().item()
            # })
            self.loss_dict[loss_name].append(loss.detach().item())
            if writer and self.iter_counter % self.count_freq == 0:
                writer.add_scalar(
                    f'loss/{loss_name}', sum(self.loss_dict[loss_name])/len(self.loss_dict[loss_name]), self.iter_counter)
        self.loss_dict['total_loss'].append(tot_loss.detach().item())
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', sum(self.loss_dict['total_loss'])/len(self.loss_dict['total_loss']), self.iter_counter)
        self.iter_counter += 1
        
        return tot_loss, loss_dict
    
    def clear_loss_dict(self):
        for loss_list in self.loss_dict.values():
            loss_list.clear()
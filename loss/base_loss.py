import torch
import torch.nn as nn
from utils.tb_wrapper import WrappedTBWriter
if 'occflow' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('occflow')
else:
    writer = None

class BaseLoss(nn.Module):

    """ Base loss class.
    args:
        weight: weight of current loss.
        input_keys: keys for actual inputs to calculate_loss().
            Since "inputs" may contain many different fields, we use input_keys
            to distinguish them.
        loss_func: the actual loss func to calculate loss.
    """

    def __init__(
            self, 
            weight=1.0,
            input_dict={
                'input': 'input'},
            start_iter=0,
            end_iter=1e10,
            **kwargs):
        super().__init__()
        self.weight = weight
        self.input_dict = input_dict
        self.loss_func = lambda: 0
        self.writer = writer
        self.start_iter = start_iter
        self.end_iter = end_iter

    # def calculate_loss(self, **kwargs):
        # return self.loss_func(*[kwargs[key] for key in self.input_keys])    

    def forward(self, inputs):
        actual_inputs = {}
        self.iter_counter = inputs['iters']
        if inputs['iters'] >= self.start_iter and inputs['iters'] < self.end_iter:
            for input_key, input_val in self.input_dict.items():
                actual_inputs.update({input_key: inputs[input_val]})
            result = self.loss_func(**actual_inputs)
            if isinstance(result, tuple):
                loss = self.weight * result[0]
                return loss, result[1]
            else:
                return self.weight * result
        else:
            loss = torch.tensor(0).cuda().float()
            return loss

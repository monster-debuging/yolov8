import copy
import torch

class EMA:
    def __init__(self, model, decay=0.9999):
        """
        指数移动平均 (Exponential Moving Average)
        model: 模型
        decay: 衰减率，通常 0.999~0.9999
        """
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        # 冻结 EMA 模型的参数，不参与梯度计算
        for param in self.ema.parameters():
            param.requires_grad = False
    
    def update(self, model):
        """
        每轮训练后更新 EMA 权重
        ema = decay * ema + (1 - decay) * model
        """
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1. - self.decay) * msd[k].detach()
# -*- coding: utf-8 -*-

import torch
torch.cuda.init()
torch.cuda.empty_cache()
from CNN_Module import Net_1
device = torch.device(0)
Model = '/srv/waveform-analysis/dataset/jinp/Nets/Channel00.torch_net'
net = Net_1().to(device)
#net = torch.load(Model, map_location=device)
print([i for i in net.parameters()])

'''
    Copyright (c) 2021 PPQ

    This file is part of pytorch learning (pl).

    pl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pl is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pl.  If not, see <https://www.gnu.org/licenses/>.
'''

import torch
import model
import dnn
import numpy as np
import PIL.Image as pi

#类似CUDA里区分CPU合GPU的方式
device = torch.device('cuda')
host = torch.device('cpu')

args = {}
args['model_type'] = 'dnn'
args['model_name'] = 'model'
args['new_model'] = 'false'
m: model.Model = dnn.getModel(**args)

for i in range(10):
    image = pi.open('data/pictures/' + str(i) + '.png').convert('L').convert('F')
    sample = torch.from_numpy(np.mat(image)).to(device).view(1, 784)
    print(m.out(sample / 255.0))

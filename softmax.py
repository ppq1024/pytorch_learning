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

import os
import torch
import numpy as np
import data
import model

from data import DataBlock

#类似CUDA里区分CPU合GPU的方式
device = torch.device('cuda')
host = torch.device('cpu')

def getModel(**args) -> model.Model:
    assert args['model_type'] == 'softmax'
    if (args['new_model'] == 'true'):
        args['model_name'] = ''
    return SoftMax(**args)

path = lambda model_name: 'models/softmax/' + model_name + '.npz'

class SoftMax(model.Model):
    def __init__(this, **args) -> None:
        args.setdefault('input_size', 784)
        args.setdefault('output_size', 10)
        args.setdefault('learning_rate', 0.03)
        super().__init__(this, **args)

    def init(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        this.__weight = torch.randn(args['input_size'], args['output_size'], dtype=torch.float32, device=device)
        this.__bias = torch.zeros(1, args['output_size'], dtype=torch.float32, device=device)
    
    def load(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        model_path = path(args['model_name'])
        model_data = np.load(model_path)
        this.__weight = torch.from_numpy(model_data['weight']).to(device)
        this.__bias = torch.from_numpy(model_data['bias']).to(device)
    
    def save(this, model_name: str) -> None:
        model_path = path(model_name)
        mats = {}
        mats['weight'] = this.__weight.to(host).numpy()
        mats['bias'] = this.__bias.to(host).numpy()
        if (not os.path.exists('models/softmax')):
            os.makedirs('models/softmax')
        np.savez(model_path, **mats)

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        iterator = data.iter(sample, label, batch_size)
        for s, l in iterator:
            y = torch.matmul(s, this.__weight) + this.__bias
            y_exp = y.exp()
            dy: torch.Tensor = (y_exp / y_exp.sum(dim=1, keepdim=True) - l) / batch_size
            this.__weight -= this.learning_rate * torch.matmul(s.T, dy)
            this.__bias -= this.learning_rate * dy.sum(dim=0)

    def test(this, sample: DataBlock, label: DataBlock) -> float:
        right = 0
        total = 0
        iterator = data.iter(sample, label, 1)
        for s, l in iterator:
            y = torch.matmul(s, this.__weight) + this.__bias
            total += 1
            if (torch.argmax(y) == torch.argmax(l)):
                right += 1
        return right / total
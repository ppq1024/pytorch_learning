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

from typing import Iterator, Tuple
import torch
import torchvision
import numpy as np
import data
from data import DataBlock
import model

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
        model.Model.__init__(this, **args)

    def init(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        this.w = torch.randn(args['input_size'], args['output_size'], dtype=torch.float32, device=device)
        this.b = torch.zeros(1, args['output_size'], dtype=torch.float32, device=device)
    
    def load(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        model_path = path(args['model_name'])
        model_data = np.load(model_path)
        this.w = torch.from_numpy(model_data['weight']).to(device)
        this.b = torch.from_numpy(model_data['bias']).to(device)
    
    def save(this, model_name: str) -> None:
        model_path = path(model_name)
        mats = {}
        mats['weight'] = this.w.to(host).numpy()
        mats['bias'] = this.b.to(host).numpy()
        np.savez(model_path, **mats)

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        iterator = data.iter(sample, label, batch_size)
        for s, l in iterator:
            y = torch.matmul(s, this.w) + this.b
            y_exp = y.exp()
            dy: torch.Tensor = (y_exp / y_exp.sum(dim=1, keepdim=True) - l) / batch_size
            this.w -= this.learning_rate * torch.matmul(s.T, dy)
            this.b -= this.learning_rate * dy.sum(dim=0)

    def test(this, sample: DataBlock, label: DataBlock) -> float:
        right = 0
        total = 0
        iterator = data.iter(sample, label, 1)
        for s, l in iterator:
            y = torch.matmul(s, this.w) + this.b
            total += 1
            if (torch.argmax(y) == torch.argmax(l)):
                right += 1
        return right / total
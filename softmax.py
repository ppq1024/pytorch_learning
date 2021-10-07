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

path = lambda model_name: 'models/softmax/' + model_name + '.pt'

class SoftMax(model.Model):
    def __init__(this, **args) -> None:
        args.setdefault('input_size', 784)
        args.setdefault('output_size', 10)
        args.setdefault('learning_rate', 0.03)
        super().__init__(**args)

    def init(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        this.__weight = torch.randn(args['input_size'], args['output_size'], dtype=torch.float32, device=device, requires_grad=True)
        this.__bias = torch.zeros(1, args['output_size'], dtype=torch.float32, device=device, requires_grad=True)
    
    def load(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        model_data = torch.load(path(args['model_name']), map_location=device)
        this.__weight: torch.Tensor = model_data['weight']
        this.__bias: torch.Tensor = model_data['bias']
    
    def save(this, model_name: str) -> None:
        if (not os.path.exists('models/softmax')):
            os.makedirs('models/softmax')
        torch.save({'weight': this.__weight, 'bias': this.__bias}, path(model_name))

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        iterator = data.iter(sample, label, batch_size)
        for s, l in iterator:
            y = torch.matmul(s, this.__weight) + this.__bias
            y_exp = y.exp()

            # dy: torch.Tensor = (y_exp / y_exp.sum(dim=1, keepdim=True) - l) / batch_size
            # this.__weight -= this.learning_rate * torch.matmul(s.T, dy)
            # this.__bias -= this.learning_rate * dy.sum(dim=0)

            loss = (torch.log(y_exp.sum(dim=1, keepdim=True)) - y) * l
            loss.sum().backward()
            this.__weight.data -= this.learning_rate * this.__weight.grad / batch_size
            this.__bias.data -= this.learning_rate * this.__bias.grad / batch_size
            this.__weight.grad.zero_()
            this.__bias.grad.zero_()


    
    def out(this, sample: torch.Tensor) -> int:
        with torch.no_grad():
            y = torch.matmul(sample, this.__weight) + this.__bias
            return torch.argmax(y).item()
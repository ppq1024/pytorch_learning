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
import numpy as np
import data
import model
import os

from data import DataBlock

#类似CUDA里区分CPU合GPU的方式
device = torch.device('cuda')
host = torch.device('cpu')

def getModel(**args) -> model.Model:
    assert args['model_type'] == 'dnn'
    if (args['new_model'] == 'true'):
        args['model_name'] = ''
    return DNN(**args)

path = lambda model_name: 'models/dnn/' + model_name + '.pt'

class DNN(model.Model):
    def __init__(this, **args) -> None:
        args.setdefault('input_size', 784)
        args.setdefault('layer_size', [512, 512, 256, 10])
        args.setdefault('learning_rate', 0.15)
        super().__init__(**args)

    def init(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        this.__weights: list[torch.Tensor] = []
        this.__biases: list[torch.Tensor] = []
        input_size = args['input_size']
        for output_size in args['layer_size']:
            this.__weights.append(torch.randn(input_size, output_size, dtype=torch.float32, device=device, requires_grad=True))
            this.__biases.append(torch.zeros(1, output_size, dtype=torch.float32, device=device, requires_grad=True))
            input_size = output_size
    
    def load(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        model_data = torch.load(path(args['model_name']))
        this.__weights: list[torch.Tensor] = model_data['weights']
        this.__biases: list[torch.Tensor] = model_data['biases']
    
    def save(this, model_name: str) -> None:
        if (not os.path.exists('models/dnn')):
            os.makedirs('models/dnn')
        torch.save({'weights': this.__weights, 'biases': this.__biases}, path(model_name))

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        iterator = data.iter(sample, label, batch_size)
        for s, l in iterator:
            # x: list[torch.Tensor] = []
            # y: list[torch.Tensor] = []
            # x.append(s)
            # layer_count = len(this.__weights)
            # for i in range(layer_count):
            #     y.append(torch.matmul(x[i], this.__weights[i]) + this.__biases[i])
            #     if (i < layer_count - 1):
            #         x.append(torch.sigmoid(y[i]))
            
            # y_exp = y[layer_count - 1].exp()
            # dy = (y_exp / y_exp.sum(dim=1, keepdim=True) - l) / batch_size
            # for i in range(layer_count - 1, -1, -1):
            #     dx = torch.matmul(dy, this.__weights[i].T)
            #     this.__weights[i] -= this.learning_rate * torch.matmul(x[i].T, dy)
            #     this.__biases[i] -= this.learning_rate * dy.sum(dim=0)
            #     if (i > 0):
            #         # dy-1 = dx * sigmoid'(x)
            #         dy = dx * x[i] * (1 - x[i])

            x = s
            layer_count = len(this.__weights)
            for i in range(layer_count):
                y = torch.matmul(x, this.__weights[i]) + this.__biases[i]
                if (i < layer_count - 1):
                    x = torch.sigmoid(y)
            y_exp = y.exp()
            loss = (torch.log(y_exp.sum(dim=1, keepdim=True)) - y) * l
            loss.sum().backward()
            for i in range(layer_count):
                this.__weights[i].data -= this.learning_rate * this.__weights[i].grad / batch_size
                this.__biases[i].data -= this.learning_rate * this.__biases[i].grad / batch_size
                this.__weights[i].grad.zero_()
                this.__biases[i].grad.zero_()
    
    def out(this, sample: torch.Tensor) -> int:
        with torch.no_grad():
            x = sample
            layer_count = len(this.__weights)
            for i in range(layer_count):
                y = torch.matmul(x, this.__weights[i]) + this.__biases[i]
                if (i < layer_count - 1):
                    x = torch.sigmoid(y)
            return torch.argmax(y).item()
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

path = lambda model_name: 'models/dnn/' + model_name + '.npz'

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
            this.__weights.append(torch.randn(input_size, output_size, dtype=torch.float32, device=device))
            this.__biases.append(torch.zeros(1, output_size, dtype=torch.float32, device=device))
            input_size = output_size
    
    def load(this, **args) -> None:
        this.learning_rate = args['learning_rate']
        this.__weights: list[torch.Tensor] = []
        this.__biases: list[torch.Tensor] = []
        model_path = path(args['model_name'])
        model_data = np.load(model_path)
        files:list[str] = model_data.files
        sorted(files)
        for file in files:
            if (file.startswith('bias')):
                this.__biases.append(torch.from_numpy(model_data[file]).to(device))
            elif (file.startswith('weight')):
                this.__weights.append(torch.from_numpy(model_data[file]).to(device))
    
    def save(this, model_name: str) -> None:
        mats = {}
        for i in range(len(this.__weights)):
            mats['weight_' + str(i)] = this.__weights[i].to(host).numpy()
        for i in range(len(this.__biases)):
            mats['bias_' + str(i)] = this.__biases[i].to(host).numpy()
        if (not os.path.exists('models/dnn')):
            os.makedirs('models/dnn')
        np.savez(path(model_name), **mats)

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        iterator = data.iter(sample, label, batch_size)
        for s, l in iterator:
            x: list[torch.Tensor] = []
            y: list[torch.Tensor] = []
            x.append(s)
            layer_count = len(this.__weights)
            for i in range(layer_count):
                y.append(torch.matmul(x[i], this.__weights[i]) + this.__biases[i])
                if (i < layer_count - 1):
                    x.append(torch.sigmoid(y[i]))
            
            y_exp = y[layer_count - 1].exp()
            dy = (y_exp / y_exp.sum(dim=1, keepdim=True) - l) / batch_size
            for i in range(layer_count - 1, -1, -1):
                dx = torch.matmul(dy, this.__weights[i].T)
                this.__weights[i] -= this.learning_rate * torch.matmul(x[i].T, dy)
                this.__biases[i] -= this.learning_rate * dy.sum(dim=0)
                if (i > 0):
                    # dy-1 = dx * sigmoid'(x)
                    dy = dx * x[i] * (1 - x[i])
    
    def out(this, sample: torch.Tensor) -> int:
        x = sample
        layer_count = len(this.__weights)
        for i in range(layer_count):
            y = torch.matmul(x, this.__weights[i]) + this.__biases[i]
            if (i < layer_count - 1):
                x = torch.sigmoid(y)
        return torch.argmax(y)
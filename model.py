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
import data
import util

from data import DataBlock

class Model:
    model_type = ''

    def __init__(this, **args) -> None:
        if (args['model_name'] == ''):
            this.init(**args)
            for param in this._model.parameters():
                torch.nn.init.normal_(param)
        else:
            this.load(**args)
        this.__loss = CrossEntropyLoss()
        this.__optimizer = torch.optim.SGD(this._model.parameters(), args['learning_rate'])

    def init(this, **args) -> None:
        pass

    def load(this, **args) -> None:
        this._model: torch.nn.Module = torch.load(util.path(args['model_type'], args['model_name']), map_location=util.active)

    def save(this, model_name:str) -> None:
        if (not os.path.exists('models/' + this.model_type)):
            os.makedirs('models/' + this.model_type)
        torch.save(this._model, util.path(this.model_type, model_name))

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        iterator = data.iter(sample, label, batch_size)
        for s, l in iterator:
            y = this._model(s)
            loss: torch.Tensor = this.__loss(y, l)
            loss = loss.sum()
            this.__optimizer.zero_grad()
            loss.backward()
            this.__optimizer.step()

    def test(this, sample: DataBlock, label: DataBlock) -> float:
        right = 0.0
        total = 0.0
        iterator = data.iter(sample, label, 1)
        for s, l in iterator:
            if (this.out(s) == torch.argmax(l).item()):
                right += 1
            total += 1
        return right / total

    def out(this, sample: torch.Tensor) -> int:
        with torch.no_grad():
            y = this._model(sample)
            return torch.argmax(y).item()

class CrossEntropyLoss(torch.nn.Module):
    def forward(this, y: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        y_exp = y.exp()
        loss = (torch.log(y_exp.sum(dim=1, keepdim=True)) - y) * l
        return loss

class Flatten(torch.nn.Module):
    def forward(this, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.shape[0], -1)
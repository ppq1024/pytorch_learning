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
import util

def getModel(**args) -> model.Model:
    assert args['model_type'] == 'cnn'
    if (args['new_model'] == 'true'):
        args['model_name'] = ''
    return CNN(**args)

class CNN(model.Model):
    model_type = 'cnn'

    def __init__(this, **args) -> None:
        args.setdefault('learning_rate', 0.03)
        super().__init__(**args)

    def init(this, **args) -> None:
        # 那么多参数懒得弄了
        this._model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, device=util.active),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5, device=util.active),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(2, 2),
            model.Flatten(),
            torch.nn.Linear(256, 256, device=util.active),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, 10, device=util.active)
        )
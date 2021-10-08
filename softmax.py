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
import util
import model

def getModel(**args) -> model.Model:
    assert args['model_type'] == 'softmax'
    if (args['new_model'] == 'true'):
        args['model_name'] = ''
    return SoftMax(**args)

class SoftMax(model.Model):
    model_type = 'softmax'

    def __init__(this, **args) -> None:
        args.setdefault('input_size', 784)
        args.setdefault('output_size', 10)
        args.setdefault('learning_rate', 0.03)
        super().__init__(**args)  

    def init(this, **args) -> None:
        this._model = torch.nn.Sequential(model.Flatten(),
            torch.nn.Linear(args['input_size'], args['output_size'], device=util.active)
        )
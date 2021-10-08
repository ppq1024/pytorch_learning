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

import util
import model

from torch import nn

def getModel(**args) -> model.Model:
    assert args['model_type'] == 'dnn'
    if (args['new_model'] == 'true'):
        args['model_name'] = ''
    return DNN(**args)

class DNN(model.Model):
    model_type = 'dnn'

    def __init__(this, **args) -> None:
        args.setdefault('input_size', 784)
        args.setdefault('layer_size', [512, 512, 256, 10])
        args.setdefault('learning_rate', 0.2)
        super().__init__(**args)

    def init(this, **args) -> None:
        
        models = []
        input_size = args['input_size']
        for output_size in args['layer_size']:
            models.append(nn.Linear(input_size, output_size, device=util.active))
            models.append(nn.Sigmoid())
            input_size = output_size
        this._model = nn.Sequential(*models)
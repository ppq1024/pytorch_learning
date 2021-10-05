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
import data
from data import DataBlock

class Model:
    def __init__(this, **args) -> None:
        if (args['model_name'] == ''):
            this.init(**args)
        else:
            this.load(**args)

    def init(this, **args) -> None:
        pass

    def load(this, **args) -> None:
        pass

    def save(this, model_name:str) -> None:
        pass

    def train(this, sample: DataBlock, label: DataBlock, batch_size = 32) -> None:
        pass

    def test(this, sample: DataBlock, label: DataBlock) -> float:
        right = 0.0
        total = 0.0
        iterator = data.iter(sample, label, 1)
        for s, l in iterator:
            if (this.out(s) == torch.argmax(l)):
                right += 1
            total += 1
        return right / total

    def out(this, sample: torch.Tensor) -> int:
        pass
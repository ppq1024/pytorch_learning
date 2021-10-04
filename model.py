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

    def save(this, model_path:str) -> None:
        pass

    def train(this, training_set:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        pass

    def test(this, testing_set:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        pass
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

from enum import Enum
from typing import Iterable, Iterator
import torch
import numpy as np

class DataType(Enum):
    undefined = 0
    sample = 1
    label = 2

class DataBlock:
    def __init__(this, data, data_type:DataType, element_length:int, size:int) -> None:
        this.data_type = data_type
        this.data:torch.Tensor = torch.tensor(data, dtype=torch.float32, device='cuda').view(size, element_length)

def load(name):
    file = open(name, 'rb', 64 * 1024)
    magic_number = int.from_bytes(file.read(4), 'big')
    size = int.from_bytes(file.read(4), 'big')
    data_type = DataType.undefined
    if (magic_number == 0x0801):
        data_type = DataType.label
    elif (magic_number == 0x0803):
        data_type = DataType.sample
    
    assert data_type != DataType.undefined
    data = []
    element_length = 10
    if (data_type == DataType.sample):
        w = int.from_bytes(file.read(4), 'big')
        h = int.from_bytes(file.read(4), 'big')
        element_length = w * h
        raw_data = file.read(element_length * size)
        file.close()
        for i in range(size):
            element = []
            for j in range(element_length):
                element.append(raw_data[i * element_length + j] / 255.0)
            data.append(element)
    elif (data_type == DataType.label):
        raw_data = file.read(size)
        file.close()
        for i in range(size):
            element = []
            for j in range(10):
                if (j == raw_data[i]):
                    element.append(1.0)
                else:
                    element.append(0.0)
            data.append(element)
    return DataBlock(data, data_type, element_length, size)

def iter(sample:DataBlock, label:DataBlock, batch_size:int = -1):
    assert sample.data_type == DataType.sample and label.data_type == DataType.label \
            and sample.data.shape[0] == label.data.shape[0]
    
    size = sample.data.shape[0]
    if (batch_size <= 0):
        batch_size = size
    for i in range(0, size, batch_size):
        yield torch.tensor(sample.data[i : min(i + batch_size, size)]), torch.tensor(label.data[i : min(i + batch_size, size)])



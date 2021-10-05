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
from enum import Enum

device = torch.device('cuda')
host = torch.device('cpu')

class DataType(Enum):
    undefined = 0
    sample = 1
    label = 2

class DataBlock:
    def __init__(this, data:torch.Tensor, data_type:DataType) -> None:
        this.data_type = data_type
        this.data = data

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
    element_length = 10
    if (data_type == DataType.sample):
        w = int.from_bytes(file.read(4), 'big')
        h = int.from_bytes(file.read(4), 'big')
        element_length = w * h
        raw_data = file.read(element_length * size)
        file.close()
        data = torch.tensor(list(raw_data), dtype=torch.float32, device=device).view(size, element_length)
        data /= 255.0
    elif (data_type == DataType.label):
        raw_data = file.read(size)
        file.close()
        data = torch.zeros(size, 10, dtype=torch.float32, device=device)
        for i in range(size):
            data[i][raw_data[i]] = 1.0
    return DataBlock(data, data_type)

def iter(sample:DataBlock, label:DataBlock, batch_size:int = -1):
    assert sample.data_type == DataType.sample and label.data_type == DataType.label \
            and sample.data.shape[0] == label.data.shape[0]
    
    size = sample.data.shape[0]
    if (batch_size <= 0):
        batch_size = size
    for i in range(0, size, batch_size):
        yield sample.data[i : min(i + batch_size, size)], label.data[i : min(i + batch_size, size)]



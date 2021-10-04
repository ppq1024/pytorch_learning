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
import torchvision
import numpy as np
import data

from data import DataBlock

device = torch.device('cuda')
host = torch.device('cpu')

def test(sample:DataBlock, label:DataBlock) -> float:
    right = 0
    total = 0
    iterator = data.iter(sample, label, 1)
    for s, l in iterator:
        y = torch.matmul(s, w) + b
        total += 1
        if (torch.argmax(y) == torch.argmax(l)):
            right += 1
    return right / total

def save(model_path):
    mats = {}
    mats['weight'] = w.to(host).numpy()
    mats['bias'] = b.to(host).numpy()
    np.savez(model_path, **mats)

def load(model_path):
    model_data = np.load(model_path)
    return torch.from_numpy(model_data['weight']).to(device), torch.from_numpy(model_data['bias']).to(device)

samples:DataBlock = data.load("data/train-data")
labels:DataBlock = data.load("data/train-label")
samples_test:DataBlock = data.load("data/test-data")
labels_test:DataBlock = data.load("data/test-label")

input_size = 784
output_size = 10
batch_size = 16
learning_rate = 0.03

init = False
if (init):
    w = torch.randn(input_size, output_size, dtype=torch.float32, device=device)
    b = torch.randn(1, output_size, dtype=torch.float32, device=device)
else:
    w, b = load('models/softmax/model.npz')

correct_rate = test(samples_test, labels_test)
print('initial correct rate:' + str(correct_rate))

o = torch.ones(1, batch_size, dtype=torch.float32, device=device)

while (correct_rate < 0.8):
    print('new loop')
    iterator = data.iter(samples, labels, batch_size)
    for s, l in iterator:
        dy = (torch.matmul(s, w) + b - l) / batch_size
        w -= learning_rate * torch.matmul(s.T, dy)
        b -= learning_rate * torch.matmul(o, dy)
    correct_rate = test(samples_test, labels_test)
    print('correct rate:' + str(correct_rate))
    save('models/softmax/model.npz')

save('models/softmax/model.npz')
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

import data
import dnn
import softmax
import cnn

from typing import Dict
from data import DataBlock
from model import Model

entance = {}

def init():
    entance['softmax'] = softmax.getModel
    entance['dnn'] = dnn.getModel
    entance['cnn'] = cnn.getModel

def launch(args:Dict[str, str]) -> None:
    init()
    args.setdefault('destination', '0.75')
    samples:DataBlock = data.load("data/train-data")
    labels:DataBlock = data.load("data/train-label")
    samples_test:DataBlock = data.load("data/test-data")
    labels_test:DataBlock = data.load("data/test-label")
    m:Model = entance[args['model_type']](**args)
    correct_rate = m.test(samples_test, labels_test)
    print('initial correct rate:' + str(correct_rate))
    while (correct_rate < float(args['destination'])):
        print('new loop')
        m.train(samples, labels)
        correct_rate = m.test(samples_test, labels_test)
        print('correct rate:' + str(correct_rate))
        m.save(args['model_name'])
    m.save(args['model_name'])
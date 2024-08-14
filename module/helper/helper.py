import math
import os
import random
import re
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import numpy as np
import torch
from geopy.distance import geodesic
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class OneTimeAssign(object):
    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError('trying to modify a constant att: {}'.format(key))

        super().__setattr__(key, value)


class IterableBase(object):
    def __init__(self):
        self._data = list()
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._data):
            result = self._data[self._index]
            self._index += 1
            return result
        else:
            self._index = 0
            raise StopIteration


class DebugHelper(object):
    DEBUG_LEVEL_BLOCK = 4
    DEBUG_LEVEL_ERROR = 3
    DEBUG_LEVEL_WARN = 2
    DEBUG_LEVEL_INFO = 1
    DEBUG_LEVEL_ALL = 0

    _debug_level = DEBUG_LEVEL_BLOCK
    _timer = dict()

    def __init__(self):
        pass

    @staticmethod
    def set_debug_level(debug_level: int):
        DebugHelper._debug_level = debug_level

    @staticmethod
    def error(message: str):
        if DebugHelper.DEBUG_LEVEL_ERROR >= DebugHelper._debug_level:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('\033[31m[{} error] {}\033[0m'.format(time_str, message))

    @staticmethod
    def warn(message: str):
        if DebugHelper.DEBUG_LEVEL_WARN >= DebugHelper._debug_level:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('\033[33m[{} warning] {}\033[0m'.format(time_str, message))

    @staticmethod
    def info(message: str):
        if DebugHelper.DEBUG_LEVEL_INFO >= DebugHelper._debug_level:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('[{} info] {}'.format(time_str, message))

    @staticmethod
    def start(timer_id: int, message: str = 'start', file_name: str = ''):
        DebugHelper._timer[timer_id] = time.time() + 8 * 3600

        if DebugHelper.DEBUG_LEVEL_INFO >= DebugHelper._debug_level:
            print('\033[34m[{} info] {}\033[0m'.format(TimeHelper.int2utc(DebugHelper._timer[timer_id]), message))

        if file_name != '':
            with open(file_name, 'a+', encoding='utf-8') as f:
                f.write('[{} info] {}\n'.format(TimeHelper.int2utc(DebugHelper._timer[timer_id]), message))

    @staticmethod
    def end(timer_id: int, message: str = 'complete', file_name: str = ''):
        start_time = DebugHelper._timer[timer_id]
        end_time = time.time() + 8 * 3600
        gap_time = int(end_time - start_time)
        DebugHelper._timer.pop(timer_id)

        if DebugHelper.DEBUG_LEVEL_INFO >= DebugHelper._debug_level:
            print('\033[34m[{} info] {}, running for {}\033[0m'.format(TimeHelper.int2utc(end_time), message, TimeHelper.second2str(gap_time)))

        if file_name != '':
            with open(file_name, 'a+', encoding='utf-8') as f:
                f.write('[{} info] {}, running for {}\n'.format(TimeHelper.int2utc(end_time), message, TimeHelper.second2str(gap_time)))

    @staticmethod
    def counter():
        number = 0
        while True:
            number += 1
            yield number


class MLHelper:
    @staticmethod
    def setup_seed(seed_value: int):
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def evaluate_performance(y_true, y_pred, average: str, digits: int = 4):
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
        recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)

        return [round(accuracy, digits) * 100, round(precision, digits) * 100, round(recall, digits) * 100, round(f1, digits) * 100]


class GeoHelper:
    # https://geopy.readthedocs.io/en/latest/#module-geopy.distance
    @staticmethod
    def get_distance(coord_1: tuple[float, float], coord_2: tuple[float, float], flag: str = 'm'):
        if flag == 'km' or flag == 'kilometers':
            return geodesic(coord_1, coord_2).kilometers
        elif flag == 'm' or flag == 'meters':
            return geodesic(coord_1, coord_2).meters
        else:
            return geodesic(coord_1, coord_2).miles

    @staticmethod
    def parse_coord(coord_str: str):
        if len(coord_str) == 0:
            return None

        coord = re.findall(r'\d+\.+\d+', coord_str)
        return tuple([float(number) for number in coord])


class TimeHelper:
    @staticmethod
    def int2utc(timestamp: float, pat: str = '%Y-%m-%d %H:%M:%S', hours: int = 0):
        dt = datetime.utcfromtimestamp(timestamp)
        dt = dt + timedelta(hours=hours)
        time_str = dt.strftime(pat)
        return time_str

    @staticmethod
    def seconds_of_day(timestamp: int, hours: int = 0):
        dt = datetime.utcfromtimestamp(timestamp)
        dt = dt + timedelta(hours=hours)
        sod = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        return (dt - sod).seconds

    @staticmethod
    def second2str(second: int):
        if second <= 0:
            return '[0]second'

        string = ''
        if second > 31536000:
            year = math.floor(second / 31536000)
            string += '[' + str(year) + ']years'
            second -= year * 31536000

        if second > 2592000:
            month = math.floor(second / 2592000)
            string += '[' + str(month) + ']months'
            second -= month * 31536000

        if second > 86400:
            day = math.floor(second / 86400)
            string += '[' + str(day) + ']days'
            second -= day * 86400

        if second > 3600:
            hour = math.floor(second / 3600)
            string += '[' + str(hour) + ']hours'
            second -= hour * 3600

        if second > 60:
            minute = math.floor(second / 60)
            string += '[' + str(minute) + ']minutes'
            second -= minute * 60

        if second > 0:
            string += '[' + str(second) + ']seconds'

        return string


class ContainerHelper:
    @staticmethod
    def get_average(lst: list, axis: int):
        lst = np.array(lst)
        avg = np.mean(lst, axis=axis)
        return tuple(avg)


class FileHelper:
    @staticmethod
    def delete_files(pof: str):
        if os.path.isfile(pof):
            os.remove(pof)
        elif os.path.isdir(pof):
            ls = os.listdir(pof)
            for i in ls:
                FileHelper.delete_files(os.path.join(pof, i))

            os.removedirs(pof)


class Predictor:
    def __init__(self, model: Module, loss_fn: Module, updater: Optimizer, train_iter: DataLoader, test_iter: DataLoader) -> None:
        super(Predictor, self).__init__()

        self._model = model
        self._model.apply(Predictor.weights_init)

        self._device = Predictor.try_gpu()
        self._loss_fn = loss_fn
        self._updater = updater
        self._train_iter = train_iter
        self._test_iter = test_iter

    @staticmethod
    def try_gpu(i: int = 0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')

        return torch.device('cpu')

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM or type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    @staticmethod
    def evaluate_correct(y_pred: tensor, y_true: tensor):
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(dim=1)

        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = y_true.argmax(dim=1)

        cmp = y_true.eq(y_pred.type(y_true.dtype))

        return int(cmp.sum())

    def train(self, num_epochs: int):
        self._model = self._model.to(self._device)
        self._model.train()

        for epoch in range(num_epochs):
            total = 0
            correct = 0

            for features, labels in self._train_iter:
                features = features.to(self._device)
                labels = labels.to(self._device)
                y_hat = self._model(features)
                loss = self._loss_fn(y_hat, labels)

                self._updater.zero_grad()
                loss.backward()
                self._updater.step()

                total += len(labels)
                correct += Predictor.evaluate_correct(y_hat, labels)

            if (epoch + 1) % 10 == 0:
                DebugHelper.info('Epoch: {}, Accuracy: {:.2f}%({}/{})'.format(epoch + 1, correct / total * 100, correct, total))

    def test(self):
        y_true = list()
        y_pred = list()
        self._model = self._model.to(self._device)
        self._model.eval()

        with torch.no_grad():
            total = 0
            correct = 0
            for features, labels in self._test_iter:
                features = features.to(self._device)
                labels = labels.to(self._device)
                y_hat = self._model(features)

                total += len(labels)
                correct += Predictor.evaluate_correct(y_hat, labels)

                y_true.append(labels)
                y_pred.append(y_hat.argmax(dim=1))

        y_true = torch.cat(y_true).tolist()
        y_pred = torch.cat(y_pred).tolist()

        return y_true, y_pred

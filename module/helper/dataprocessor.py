import os
import sys
from typing import Union

import numpy as np

from module.config.config import Constant, ExperimentStage, Config
from module.helper.filereader import FileReader
from module.helper.helper import DebugHelper
from module.method.event import EventSet
from module.method.eventlet import Eventlet
from module.method.poi import POISet, POI
from module.method.stop import StopSet


class DataProcessor:
    @staticmethod
    def set_config(data_set: str, **kwargs):
        time_th = kwargs.get('time_th', Constant.BEST_TIME_TH)
        block_size = kwargs.get('block_size', Constant.BEST_BLOCK_SIZE)

        if data_set != Config.data_set or time_th != Config.time_th or block_size != Config.block_size:
            FileReader.clear_gps_data()
            FileReader.clear_stop_set()
            FileReader.clear_stop_seq()
            FileReader.clear_event_set()
            FileReader.clear_event_seq()
            Config.set_config(data_set, time_th, block_size)

    @staticmethod
    def data_preprocessing():
        assert Config.block_size <= Config.time_th
        if not os.path.exists(Config.stop_set_path):
            s_step = 0
            e_step = 0
        else:
            if os.path.exists(Config.stop_seq_path):
                s_step = 2
            else:
                s_step = 1

            if os.path.exists(Config.event_seq_path):
                e_step = 3
            elif os.path.exists(Config.event_set_path):
                e_step = 2
            else:
                e_step = 1

        s_prefix = '{}_{}_{}'.format(Constant.DATA_SSEQ, Config.data_set, Config.time_th)
        e_prefix = '{}_{}_{}'.format(Constant.DATA_ESEQ, Config.data_set, Config.time_th)

        if s_step == 0 or e_step == 0:
            DebugHelper.info('[{}] start with s_step 0/2'.format(s_prefix))
            DebugHelper.info('[{}] start with e_step 0/3'.format(e_prefix))
            user_train_data_gps = FileReader.get_user_train_data_gps()
            user_test_data_gps = FileReader.get_user_test_data_gps()
            user_train_data_date = FileReader.get_user_train_data_date()
            user_test_data_date = FileReader.get_user_test_data_date()
            user_train_data_seg = FileReader.get_user_train_data_seg()
            user_test_data_seg = FileReader.get_user_test_data_seg()
            user_train_data_emo = FileReader.get_user_train_data_emo()
            user_test_data_emo = FileReader.get_user_test_data_emo()
            Eventlet.get_stop_set(user_train_data_gps, user_train_data_date, user_train_data_seg, user_train_data_emo, ExperimentStage.TRAIN)
            Eventlet.get_stop_set(user_test_data_gps, user_test_data_date, user_test_data_seg, user_test_data_emo, ExperimentStage.TEST)
            s_step = 1
            e_step = 1

        if s_step == 1 or e_step == 1:
            user_train_stop_set = FileReader.get_user_train_stop_set()
            user_test_stop_set = FileReader.get_user_test_stop_set()
            user_train_data_date = FileReader.get_user_train_data_date()
            user_test_data_date = FileReader.get_user_test_data_date()
            user_train_data_seg = FileReader.get_user_train_data_seg()
            user_test_data_seg = FileReader.get_user_test_data_seg()
            user_train_data_emo = FileReader.get_user_train_data_emo()
            user_test_data_emo = FileReader.get_user_test_data_emo()

            if s_step == 1:
                DebugHelper.info('[{}] start with s_step 1/2'.format(s_prefix))
                Eventlet.get_stop_seq(user_train_stop_set, user_train_data_date, user_train_data_seg, user_train_data_emo, ExperimentStage.TRAIN)
                Eventlet.get_stop_seq(user_test_stop_set, user_test_data_date, user_test_data_seg, user_test_data_emo, ExperimentStage.TEST)
                s_step = 2

            if e_step == 1:
                DebugHelper.info('[{}] start with e_step 1/3'.format(e_prefix))
                poi_set = FileReader.get_poi_set()
                POISet.set_base_poi_set(poi_set)
                Eventlet.get_event_set(user_train_stop_set, user_train_data_date, user_train_data_seg, user_train_data_emo, ExperimentStage.TRAIN)
                Eventlet.get_event_set(user_test_stop_set, user_test_data_date, user_test_data_seg, user_test_data_emo, ExperimentStage.TEST)
                e_step = 2

        if e_step == 2:
            DebugHelper.info('[{}] start with e_step 2/3'.format(e_prefix))
            user_train_event_set = FileReader.get_user_train_event_set()
            user_test_event_set = FileReader.get_user_test_event_set()
            user_train_data_date = FileReader.get_user_train_data_date()
            user_test_data_date = FileReader.get_user_test_data_date()
            user_train_data_seg = FileReader.get_user_train_data_seg()
            user_test_data_seg = FileReader.get_user_test_data_seg()
            user_train_data_emo = FileReader.get_user_train_data_emo()
            user_test_data_emo = FileReader.get_user_test_data_emo()
            Eventlet.get_event_seq(user_train_event_set, user_train_data_date, user_train_data_seg, user_train_data_emo, ExperimentStage.TRAIN)
            Eventlet.get_event_seq(user_test_event_set, user_test_data_date, user_test_data_seg, user_test_data_emo, ExperimentStage.TEST)
            e_step = 3

        if s_step == 2:
            DebugHelper.info('[{}] data is ready'.format(s_prefix))

        if e_step == 3:
            DebugHelper.info('[{}] data is ready'.format(e_prefix))

    @staticmethod
    def set_load(is_stop: bool, save: bool):
        prefix = '{}_{}_{}'.format(Constant.DATA_SSET if is_stop else Constant.DATA_ESET, Config.data_set, Config.time_th)
        try:
            train_x = np.load(Config.npy_data_path + prefix + '_train_x.npy')
            train_y = np.load(Config.npy_data_path + prefix + '_train_y.npy')
            test_x = np.load(Config.npy_data_path + prefix + '_test_x.npy')
            test_y = np.load(Config.npy_data_path + prefix + '_test_y.npy')
            DebugHelper.info('[{}] data shape: train[{}], test[{}]'.format(prefix, np.shape(train_x), np.shape(test_x)))
            return train_x, train_y, test_x, test_y
        except FileNotFoundError:
            DebugHelper.info('missing npy data, start preparing...')

        if is_stop:
            user_train_set = FileReader.get_user_train_stop_set()
            user_test_set = FileReader.get_user_test_stop_set()
        else:
            user_train_set = FileReader.get_user_train_event_set()
            user_test_set = FileReader.get_user_test_event_set()

        user_train_seg = FileReader.get_user_train_data_seg()
        user_test_seg = FileReader.get_user_test_data_seg()
        user_train_emo = FileReader.get_user_train_data_emo()
        user_test_emo = FileReader.get_user_test_data_emo()

        data_sets = {
            ExperimentStage.TRAIN: [user_train_set, user_train_seg, user_train_emo],
            ExperimentStage.TEST: [user_test_set, user_test_seg, user_test_emo],
        }

        data_x = list()
        data_y = list()
        data_len = 0
        train_size = 0
        for stage, data_set in data_sets.items():
            for user_id, es_set, segment, emotion in zip(data_set[0].keys(), data_set[0].values(), data_set[1].values(), data_set[2].values()):
                sample = DataProcessor._get_set_sample(user_id, segment, es_set)
                data_x.append(sample)
                data_y.append(emotion - 1)
                data_len = max(data_len, len(sample))
                if stage == ExperimentStage.TRAIN:
                    train_size += 1

        for d in data_x:
            gap = data_len - len(d)
            for i in range(gap):
                d.append(d[-1])

        train_x = np.array(data_x[:train_size])
        train_y = np.array(data_y[:train_size])
        test_x = np.array(data_x[train_size:])
        test_y = np.array(data_y[train_size:])

        if save:
            fpath = Config.npy_data_path
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            np.save(fpath + prefix + '_train_x.npy', train_x)
            np.save(fpath + prefix + '_train_y.npy', train_y)
            np.save(fpath + prefix + '_test_x.npy', test_x)
            np.save(fpath + prefix + '_test_y.npy', test_y)

        DebugHelper.info('[{}] data shape: train[{}], test[{}]'.format(prefix, np.shape(train_x), np.shape(test_x)))
        return train_x, train_y, test_x, test_y

    @staticmethod
    def seq_load(is_stop: bool, save: bool, exp_type: str):
        prefix = '{}_{}_{}_{}_{}'.format(Constant.DATA_SSEQ if is_stop else Constant.DATA_ESEQ, Config.data_set, Config.time_th, Config.block_size, exp_type)
        try:
            train_x = np.load(Config.npy_data_path + prefix + '_train_x.npy')
            train_y = np.load(Config.npy_data_path + prefix + '_train_y.npy')
            test_x = np.load(Config.npy_data_path + prefix + '_test_x.npy')
            test_y = np.load(Config.npy_data_path + prefix + '_test_y.npy')
            DebugHelper.info('[{}] data shape: train[{}], test[{}]'.format(prefix, np.shape(train_x), np.shape(test_x)))
            return train_x, train_y, test_x, test_y
        except FileNotFoundError:
            DebugHelper.info('missing npy data, start preparing...')

        if is_stop:
            user_train_seq = FileReader.get_user_train_stop_seq()
            user_test_seq = FileReader.get_user_test_stop_seq()
        else:
            user_train_seq = FileReader.get_user_train_event_seq()
            user_test_seq = FileReader.get_user_test_event_seq()

        user_train_seg = FileReader.get_user_train_data_seg()
        user_test_seg = FileReader.get_user_test_data_seg()
        user_train_emo = FileReader.get_user_train_data_emo()
        user_test_emo = FileReader.get_user_test_data_emo()

        data_sets = {
            ExperimentStage.TRAIN: [user_train_seq, user_train_seg, user_train_emo],
            ExperimentStage.TEST: [user_test_seq, user_test_seg, user_test_emo],
        }

        data_x = list()
        data_y = list()
        train_size = 0
        for stage, data_set in data_sets.items():
            for user_id, es_seq, segment, emotion in zip(data_set[0].keys(), data_set[0].values(), data_set[1].values(), data_set[2].values()):
                sample = DataProcessor._get_seq_sample(user_id, segment, es_seq)
                data_x.append(sample)
                data_y.append(emotion - 1)
                if stage == ExperimentStage.TRAIN:
                    train_size += 1

        train_x = np.array(data_x[:train_size])
        train_y = np.array(data_y[:train_size])
        test_x = np.array(data_x[train_size:])
        test_y = np.array(data_y[train_size:])

        if save:
            fpath = Config.npy_data_path
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            np.save(fpath + prefix + '_train_x.npy', train_x)
            np.save(fpath + prefix + '_train_y.npy', train_y)
            np.save(fpath + prefix + '_test_x.npy', test_x)
            np.save(fpath + prefix + '_test_y.npy', test_y)

        DebugHelper.info('[{}] data shape: train[{}], test[{}]'.format(prefix, np.shape(train_x), np.shape(test_x)))
        return train_x, train_y, test_x, test_y

    @staticmethod
    def _get_set_sample(user_id: int, segment: int, es_set: Union[StopSet, EventSet]):
        sample = list()

        for es_vec in es_set.to_vector():
            es_vec[0] -= int(es_vec[0])
            es_vec[1] -= int(es_vec[1])
            es_vec[2] %= Constant.CYCLE
            es_vec[3] %= Constant.CYCLE
            if isinstance(es_set, EventSet):
                es_vec[4] /= 100

            if user_id > -1:
                es_vec.append(user_id / 1000)

            es_vec.append(segment / 10)
            sample.append(es_vec)

        return sample

    @staticmethod
    def _get_seq_sample(user_id: int, segment: int, es_seq: list):
        sample = list()

        for es in es_seq:
            if isinstance(es, tuple):
                es_vec = list(es)
                es_vec[0] -= int(es_vec[0])
                es_vec[1] -= int(es_vec[1])
            elif isinstance(es, POI):
                es_vec = es.to_vector()
                es_vec[0] /= 100
                es_vec[1] -= int(es_vec[1])
                es_vec[2] -= int(es_vec[2])
            else:
                DebugHelper.error('error sequence type')
                sys.exit(1)

            if user_id > -1:
                es_vec.append(user_id / 1000)

            es_vec.append(segment / 10)
            sample.append(es_vec)

        return sample

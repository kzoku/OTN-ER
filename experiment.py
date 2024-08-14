import argparse
from argparse import Namespace

import numpy as np
import torch
import torch.utils.data as tud
from torch import nn
from torch.utils import data

from module.config.config import Constant, ExperimentType, Config
from module.helper.dataprocessor import DataProcessor
from module.helper.filewriter import FileWriter
from module.helper.helper import DebugHelper, Predictor, MLHelper, FileHelper
from module.method.comparison import TimeSeriesModel, BaselineModel, SOTAModel
from module.method.method import EmotionPredictModel


class TestEmotion:
    def __init__(self) -> None:
        super(TestEmotion, self).__init__()

    @staticmethod
    def test(args: Namespace):
        if args.data_type == Constant.DATA_SSET:
            train_x, train_y, test_x, test_y = DataProcessor.set_load(True, True)
            file_name = '{}_{}'.format(args.clf_type, Config.time_th)
        elif args.data_type == Constant.DATA_SSEQ:
            train_x, train_y, test_x, test_y = DataProcessor.seq_load(True, True, args.exp_type)
            file_name = '{}_{}_{}'.format(args.clf_type, Config.time_th, Config.block_size)
        elif args.data_type == Constant.DATA_ESET:
            train_x, train_y, test_x, test_y = DataProcessor.set_load(False, True)
            file_name = '{}_{}'.format(args.clf_type, Config.time_th)
        elif args.data_type == Constant.DATA_ESEQ:
            train_x, train_y, test_x, test_y = DataProcessor.seq_load(False, True, args.exp_type)
            file_name = '{}_{}_{}'.format(args.clf_type, Config.time_th, Config.block_size)
        else:
            exit(-1)

        data_len = train_x.shape[-2]
        data_size = train_x.shape[-1]
        un1 = list(np.unique(train_y))
        un2 = list(np.unique(test_y))
        n_class = len(list(np.unique(un1 + un2)))

        if args.clf_type in Constant.BL_METHODS:
            model = BaselineModel(args.clf_type, n_class)
        elif args.clf_type in Constant.TS_METHODS:
            model = TimeSeriesModel(data_size, data_len, n_class, args)
        elif args.clf_type in Constant.ST_METHODS:
            model = SOTAModel(args.clf_type, n_class)
        else:
            model = EmotionPredictModel(data_size, n_class, args)

        if args.batch_size == 0:
            args.batch_size = 16
            suit_batch_size = train_x.shape[0] // 20
            while args.batch_size < suit_batch_size:
                args.batch_size *= 2

        if args.clf_type in Constant.TS_METHODS:
            predictor = Predictor(
                model,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(model.parameters(), lr=args.learning_rate),
                data.DataLoader(tud.TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)), args.batch_size, shuffle=True),
                data.DataLoader(tud.TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long)), args.batch_size, shuffle=False)
            )

            predictor.train(args.n_epochs)
            y_true, y_pred = predictor.test()
        else:
            y_true = test_y
            y_pred = model(train_x, train_y, test_x)

        FileWriter.record_prediction(y_true, y_pred, file_name)
        FileWriter.add_result(args.clf_type, MLHelper.evaluate_performance(y_true, y_pred, 'macro'))
        torch.cuda.empty_cache()


class Experiment:
    _TIMER_ID = next(DebugHelper.counter())

    @staticmethod
    def initialization():
        MLHelper.setup_seed(200)
        DebugHelper.set_debug_level(DebugHelper.DEBUG_LEVEL_ALL)
        DebugHelper.info('torch version: {}'.format(torch.__version__))
        DebugHelper.info('torch available: {}'.format(torch.cuda.is_available()))
        FileWriter.initialization()

    @staticmethod
    def _initialization(data_set: str, exp_type: str):
        DebugHelper.info('data set: {}, experiment type: {}'.format(data_set, exp_type))

        Config.set_log_path(data_set, exp_type)
        FileHelper.delete_files(Config.exp_log_path)

    @staticmethod
    def experiment_effectiveness():
        for data_set in Constant.DATA_SETS:
            Experiment._initialization(data_set, ExperimentType.EFFECTIVENESS)
            DataProcessor.set_config(data_set)
            DataProcessor.data_preprocessing()

            FileWriter.clear()
            for clf_type in Constant.BL_METHODS + Constant.TS_METHODS + Constant.ST_METHODS:
                args = Experiment._parse_args(data_set, Constant.DATA_ESEQ, clf_type, ExperimentType.EFFECTIVENESS)
                DebugHelper.start(Experiment._TIMER_ID, '[{}, {}] start'.format(data_set, clf_type))
                TestEmotion.test(args)
                DebugHelper.end(Experiment._TIMER_ID, '[{}, {}] complete'.format(data_set, clf_type))

            FileWriter.record_result(data_set)

    @staticmethod
    def experiment_time_th():
        for data_set in Constant.DATA_SETS:
            Experiment._initialization(data_set, ExperimentType.TIME_TH)
            for time_th in Constant.TIME_THS:
                if time_th < Config.block_size:
                    continue

                DataProcessor.set_config(data_set, time_th=time_th)
                DataProcessor.data_preprocessing()

                FileWriter.clear()
                for clf_type in Constant.BL_METHODS + Constant.TS_METHODS + Constant.ST_METHODS:
                    args = Experiment._parse_args(data_set, Constant.DATA_ESEQ, clf_type, ExperimentType.TIME_TH)
                    DebugHelper.start(Experiment._TIMER_ID, '[{}, {}] start'.format(data_set, clf_type))
                    TestEmotion.test(args)
                    DebugHelper.end(Experiment._TIMER_ID, '[{}, {}] complete'.format(data_set, clf_type))

                FileWriter.record_result(time_th)

    @staticmethod
    def experiment_block_size():
        for data_set in Constant.DATA_SETS:
            Experiment._initialization(data_set, ExperimentType.BLOCK_SIZE)
            for block_size in Constant.BLOCK_SIZES:
                if block_size > Config.time_th:
                    continue

                DataProcessor.set_config(data_set, block_size=block_size)
                DataProcessor.data_preprocessing()

                FileWriter.clear()
                for clf_type in Constant.BL_METHODS + Constant.TS_METHODS + Constant.ST_METHODS:
                    args = Experiment._parse_args(data_set, Constant.DATA_ESEQ, clf_type, ExperimentType.BLOCK_SIZE)
                    DebugHelper.start(Experiment._TIMER_ID, '[{}, {}] start'.format(data_set, clf_type))
                    TestEmotion.test(args)
                    DebugHelper.end(Experiment._TIMER_ID, '[{}, {}] complete'.format(data_set, clf_type))

                FileWriter.record_result(block_size)

    @staticmethod
    def experiment_data_type():
        for data_set in Constant.DATA_SETS:
            Experiment._initialization(data_set, ExperimentType.DATA_TYPE)
            for data_type in Constant.DATA_TYPES:
                DataProcessor.set_config(data_set)
                DataProcessor.data_preprocessing()

                FileWriter.clear()
                for clf_type in Constant.BL_METHODS + Constant.TS_METHODS + Constant.ST_METHODS:
                    args = Experiment._parse_args(data_set, data_type, clf_type, ExperimentType.DATA_TYPE)
                    DebugHelper.start(Experiment._TIMER_ID, '[{}, {}] start'.format(data_set, clf_type))
                    TestEmotion.test(args)
                    DebugHelper.end(Experiment._TIMER_ID, '[{}, {}] complete'.format(data_set, clf_type))

                FileWriter.record_result(data_type)

    @staticmethod
    def _parse_args(data_set: str, data_type: str, clf_type: str, exp_type: str = ''):
        parser = argparse.ArgumentParser(description="Parameters of Model.")
        parser.add_argument('--data_set', type=str, default=data_set)
        parser.add_argument('--data_type', type=str, default=data_type)
        parser.add_argument('--clf_type', type=str, default=clf_type)
        parser.add_argument('--exp_type', type=str, default=exp_type)
        parser.add_argument('--batch_size', type=int, default=0)
        parser.add_argument('--n_hidden', type=int, default=64)
        parser.add_argument('--n_epochs', type=int, default=1500)
        parser.add_argument('--learning_rate', type=int, default=0.001)

        if clf_type in [Constant.RNN, Constant.GRU, Constant.LSTM]:
            parser.add_argument('--n_layers', type=int, default=4)
        elif clf_type == Constant.Transformer:
            parser.add_argument('--n_layers', type=int, default=2)
            parser.add_argument('--n_heads', type=int, default=1)

        return parser.parse_args()

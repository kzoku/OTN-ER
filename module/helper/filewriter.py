import os

import pandas as pd
from sklearn import metrics

from module.config.config import Constant, Config
from module.helper.helper import FileHelper
from module.method.event import EventSet
from module.method.stop import StopSet


class FileWriter:
    _result = dict()
    _result['Methods'] = list()
    _result['Accuracy'] = list()
    _result['Precision'] = list()
    _result['Recall'] = list()
    _result['F1'] = list()

    @staticmethod
    def initialization():
        FileWriter.record_unknown_event(None, True)

    @staticmethod
    def clear():
        FileWriter._result['Methods'].clear()
        FileWriter._result['Accuracy'].clear()
        FileWriter._result['Precision'].clear()
        FileWriter._result['Recall'].clear()
        FileWriter._result['F1'].clear()

    @staticmethod
    def record_unknown_event(events: [list, None], clear: bool):
        file_path = Constant.POISET_PATH
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file = '{}NEW_POI.txt'.format(file_path)
        if clear:
            FileHelper.delete_files(file)
            return

        if events is not None and len(events) > 0:
            with open(file, 'a+', encoding='utf-8') as f:
                for event in events:
                    f.write(event.to_string() + '\n')

    @staticmethod
    def add_result(clf_type: str, result: list):
        FileWriter._result['Methods'].append(clf_type)
        FileWriter._result['Accuracy'].append(result[0])
        FileWriter._result['Precision'].append(result[1])
        FileWriter._result['Recall'].append(result[2])
        FileWriter._result['F1'].append(result[3])

    @staticmethod
    def record_result(file_name: str):
        file_path = Config.exp_log_path
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = '{}.csv'.format(file_name)
        dataframe = pd.DataFrame(FileWriter._result)
        dataframe.to_csv(file_path + file_name, index=False, sep=',')

    @staticmethod
    def record_prediction(y_true, y_pred, file_name: str):
        file_path = Config.exp_log_path
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        with open(file_path + file_name + '.txt', 'w', encoding='utf-8') as f:
            f.write('Confusion Matrix:\n' + str(metrics.confusion_matrix(y_true, y_pred)) + '\n')
            f.write('Predict Result:\n' + str(metrics.classification_report(y_true, y_pred, digits=4)) + '\n')

    @staticmethod
    def record_stop_set(stop_set: StopSet, user_id: int, stage: str, date_str: str, segment: int, emotion: int):
        file_path = '{}{}/{}/'.format(Config.stop_set_path, user_id, stage)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = '{}-{}-{}.txt'.format(date_str, segment, emotion)
        with open(file_path + file_name, 'w', encoding='utf-8') as f:
            f.write(str(stop_set))

    @staticmethod
    def record_stop_seq(stop_sequence: list, user_id: int, stage: str, date_str: str, segment: int, emotion: int):
        file_path = '{}{}/{}/'.format(Config.stop_seq_path, user_id, stage)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = '{}-{}-{}.txt'.format(date_str, segment, emotion)
        with open(file_path + file_name, 'w', encoding='utf-8') as f:
            for center in stop_sequence:
                f.write(str(center) + '\n')

    @staticmethod
    def record_event_set(event_set: EventSet, user_id: int, stage: str, date_str: str, segment: int, emotion: int):
        file_path = '{}{}/{}/'.format(Config.event_set_path, user_id, stage)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = '{}-{}-{}'.format(date_str, segment, emotion)
        with open(file_path + file_name + '-read.txt', 'w', encoding='utf-8') as f:
            f.write(str(event_set))
        with open(file_path + file_name + '-parse.txt', 'w', encoding='utf-8') as f:
            f.write(event_set.to_string())

    @staticmethod
    def record_event_seq(event_sequence: list, user_id: int, stage: str, date_str: str, segment: int, emotion: int):
        file_path = '{}{}/{}/'.format(Config.event_seq_path, user_id, stage)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = '{}-{}-{}.txt'.format(date_str, segment, emotion)
        with open(file_path + file_name, 'w', encoding='utf-8') as f:
            for poi in event_sequence:
                f.write(str(poi) + '\n')

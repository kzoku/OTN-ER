import os

from module.config.config import Constant, ExperimentStage, Config
from module.helper.helper import DebugHelper, GeoHelper
from module.method.event import EventSet
from module.method.poi import POI, POISet
from module.method.stop import StopSet
from module.method.trajectory import Point


class FileReader:
    _poi_set = list()
    _user_train_data_gps = dict()
    _user_test_data_gps = dict()
    _user_train_data_emo = dict()
    _user_test_data_emo = dict()
    _user_train_data_date = dict()
    _user_test_data_date = dict()
    _user_train_data_seg = dict()
    _user_test_data_seg = dict()
    _user_train_stop_set = dict()
    _user_test_stop_set = dict()
    _user_train_stop_seq = dict()
    _user_test_stop_seq = dict()
    _user_train_event_set = dict()
    _user_test_event_set = dict()
    _user_train_event_seq = dict()
    _user_test_event_seq = dict()

    def __init__(self):
        pass

    @staticmethod
    def clear_gps_data():
        FileReader._user_train_data_gps.clear()
        FileReader._user_test_data_gps.clear()

    @staticmethod
    def clear_stop_set():
        FileReader._user_train_stop_set.clear()
        FileReader._user_test_stop_set.clear()

    @staticmethod
    def clear_stop_seq():
        FileReader._user_train_stop_seq.clear()
        FileReader._user_test_stop_seq.clear()

    @staticmethod
    def clear_event_set():
        FileReader._user_train_event_set.clear()
        FileReader._user_test_event_set.clear()

    @staticmethod
    def clear_event_seq():
        FileReader._user_train_event_seq.clear()
        FileReader._user_test_event_seq.clear()

    @staticmethod
    def get_poi_set(read_force: bool = False):
        if read_force or len(FileReader._poi_set) == 0:
            FileReader._read_poi_set()

        return FileReader._poi_set

    @staticmethod
    def get_user_train_data_gps(read_force: bool = False):
        if read_force or len(FileReader._user_train_data_gps) == 0:
            FileReader._read_user_data_gps()

        return FileReader._user_train_data_gps

    @staticmethod
    def get_user_test_data_gps(read_force: bool = False):
        if read_force or len(FileReader._user_test_data_gps) == 0:
            FileReader._read_user_data_gps()

        return FileReader._user_test_data_gps

    @staticmethod
    def get_user_train_data_date():
        assert len(FileReader._user_train_data_date) > 0
        return FileReader._user_train_data_date

    @staticmethod
    def get_user_test_data_date():
        assert len(FileReader._user_test_data_date) > 0
        return FileReader._user_test_data_date

    @staticmethod
    def get_user_train_data_seg():
        assert len(FileReader._user_train_data_seg) > 0
        return FileReader._user_train_data_seg

    @staticmethod
    def get_user_test_data_seg():
        assert len(FileReader._user_test_data_seg) > 0
        return FileReader._user_test_data_seg

    @staticmethod
    def get_user_train_data_emo():
        assert len(FileReader._user_train_data_emo) > 0
        return FileReader._user_train_data_emo

    @staticmethod
    def get_user_test_data_emo():
        assert len(FileReader._user_test_data_emo) > 0
        return FileReader._user_test_data_emo

    @staticmethod
    def get_user_train_stop_set(read_force: bool = False):
        if read_force or len(FileReader._user_train_stop_set) == 0:
            FileReader._read_user_stop_set()

        return FileReader._user_train_stop_set

    @staticmethod
    def get_user_test_stop_set(read_force: bool = False):
        if read_force or len(FileReader._user_test_stop_set) == 0:
            FileReader._read_user_stop_set()

        return FileReader._user_test_stop_set

    @staticmethod
    def get_user_train_stop_seq(read_force: bool = False):
        if read_force or len(FileReader._user_train_stop_seq) == 0:
            FileReader._read_user_stop_seq()

        return FileReader._user_train_stop_seq

    @staticmethod
    def get_user_test_stop_seq(read_force: bool = False):
        if read_force or len(FileReader._user_test_stop_seq) == 0:
            FileReader._read_user_stop_seq()

        return FileReader._user_test_stop_seq

    @staticmethod
    def get_user_train_event_set(read_force: bool = False):
        if read_force or len(FileReader._user_train_event_set) == 0:
            FileReader._read_user_event_set()

        return FileReader._user_train_event_set

    @staticmethod
    def get_user_test_event_set(read_force: bool = False):
        if read_force or len(FileReader._user_test_event_set) == 0:
            FileReader._read_user_event_set()

        return FileReader._user_test_event_set

    @staticmethod
    def get_user_train_event_seq(read_force: bool = False):
        if read_force or len(FileReader._user_train_event_seq) == 0:
            FileReader._read_user_event_seq()

        return FileReader._user_train_event_seq

    @staticmethod
    def get_user_test_event_seq(read_force: bool = False):
        if read_force or len(FileReader._user_test_event_seq) == 0:
            FileReader._read_user_event_seq()

        return FileReader._user_test_event_seq

    @staticmethod
    def _read_poi_set():
        FileReader._poi_set.clear()

        file_path: str = Constant.POISET_PATH
        with open(file_path + 'XDU_POI.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    DebugHelper.info('POI[' + line[1:-1] + '] is ignored')
                    continue

                info = line[:-1].split(',')
                name = info[0]
                coord = (float(info[1]), float(info[2]))
                ptype = int(info[3])
                FileReader._poi_set.append(POI(name, coord, ptype))

    @staticmethod
    def _read_user_data_gps():
        FileReader.clear_gps_data()
        FileReader._clear_esd()

        file_path: str = Config.gps_data_path
        user_ids = os.listdir(file_path)
        for user_id in user_ids:
            stages = os.listdir(file_path + user_id)
            for stage in stages:
                gps_data = list()
                emotions = list()
                segments = list()
                dates = list()

                files = os.listdir(file_path + user_id + '/' + stage)
                for file in files:
                    date, segment, emotion, _ = FileReader._parse_file(file)

                    points = list()
                    with open(file_path + user_id + '/' + stage + '/' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            info = line[:-1].split(',')
                            timestamp = int(info[0])
                            latitude = float(info[1])
                            longitude = float(info[2])
                            coord = (latitude, longitude)
                            points.append(Point(coord, timestamp))

                    if len(points) > 0:
                        gps_data.append(points)
                        emotions.append(emotion)
                        segments.append(segment)
                        dates.append(date)
                    else:
                        DebugHelper.warn('empty gps data, file: {}{}/{}/{}'.format(file_path, user_id, stage, file))

                if stage == ExperimentStage.TRAIN:
                    FileReader._user_train_data_gps[int(user_id)] = gps_data
                    FileReader._user_train_data_emo[int(user_id)] = emotions
                    FileReader._user_train_data_seg[int(user_id)] = segments
                    FileReader._user_train_data_date[int(user_id)] = dates
                elif stage == ExperimentStage.TEST:
                    FileReader._user_test_data_gps[int(user_id)] = gps_data
                    FileReader._user_test_data_emo[int(user_id)] = emotions
                    FileReader._user_test_data_seg[int(user_id)] = segments
                    FileReader._user_test_data_date[int(user_id)] = dates

    @staticmethod
    def _read_user_stop_set():
        FileReader.clear_stop_set()
        FileReader._clear_esd()

        file_path: str = Config.stop_set_path
        user_ids = os.listdir(file_path)
        for user_id in user_ids:
            stages = os.listdir(file_path + user_id)
            for stage in stages:
                stop_sets = list()
                emotions = list()
                segments = list()
                dates = list()

                files = os.listdir(file_path + user_id + '/' + stage)
                for file in files:
                    date, segment, emotion, _ = FileReader._parse_file(file)

                    stop_set = StopSet()
                    with open(file_path + user_id + '/' + stage + '/' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            stop = StopSet.create_stop(line[:-1])
                            if stop is not None:
                                stop_set.add_stop(stop)

                    if len(stop_set) > 0:
                        stop_sets.append(stop_set)
                        emotions.append(emotion)
                        segments.append(segment)
                        dates.append(date)
                    else:
                        DebugHelper.warn('empty stop set, file: {}{}/{}/{}'.format(file_path, user_id, stage, file))

                if stage == ExperimentStage.TRAIN:
                    FileReader._user_train_stop_set[int(user_id)] = stop_sets
                    FileReader._user_train_data_emo[int(user_id)] = emotions
                    FileReader._user_train_data_seg[int(user_id)] = segments
                    FileReader._user_train_data_date[int(user_id)] = dates
                elif stage == ExperimentStage.TEST:
                    FileReader._user_test_stop_set[int(user_id)] = stop_sets
                    FileReader._user_test_data_emo[int(user_id)] = emotions
                    FileReader._user_test_data_seg[int(user_id)] = segments
                    FileReader._user_test_data_date[int(user_id)] = dates

    @staticmethod
    def _read_user_stop_seq():
        FileReader.clear_stop_seq()
        FileReader._clear_esd()

        file_path: str = Config.stop_seq_path
        user_ids = os.listdir(file_path)
        for user_id in user_ids:
            stages = os.listdir(file_path + user_id)
            for stage in stages:
                stop_sequences = list()
                emotions = list()
                segments = list()
                dates = list()

                files = os.listdir(file_path + user_id + '/' + stage)
                for file in files:
                    date, segment, emotion, _ = FileReader._parse_file(file)

                    stop_sequence = list()
                    with open(file_path + user_id + '/' + stage + '/' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            coord = GeoHelper.parse_coord(line[:-1])
                            if coord is not None:
                                stop_sequence.append(coord)

                    if len(stop_sequence) > 0:
                        stop_sequences.append(stop_sequence)
                        emotions.append(emotion)
                        segments.append(segment)
                        dates.append(date)
                    else:
                        DebugHelper.warn('empty stop sequence, file: {}{}/{}/{}'.format(file_path, user_id, stage, file))

                if stage == ExperimentStage.TRAIN:
                    FileReader._user_train_stop_seq[int(user_id)] = stop_sequences
                    FileReader._user_train_data_emo[int(user_id)] = emotions
                    FileReader._user_train_data_seg[int(user_id)] = segments
                    FileReader._user_train_data_date[int(user_id)] = dates
                elif stage == ExperimentStage.TEST:
                    FileReader._user_test_stop_seq[int(user_id)] = stop_sequences
                    FileReader._user_test_data_emo[int(user_id)] = emotions
                    FileReader._user_test_data_seg[int(user_id)] = segments
                    FileReader._user_test_data_date[int(user_id)] = dates

    @staticmethod
    def _read_user_event_set():
        FileReader.clear_event_set()
        FileReader._clear_esd()

        file_path: str = Config.event_set_path
        user_ids = os.listdir(file_path)
        for user_id in user_ids:
            stages = os.listdir(file_path + user_id)
            for stage in stages:
                event_sets = list()
                segments = list()
                emotions = list()
                dates = list()

                files = os.listdir(file_path + user_id + '/' + stage)
                for file in files:
                    date, segment, emotion, is_read = FileReader._parse_file(file)
                    if not is_read:
                        continue

                    event_set = EventSet()
                    with open(file_path + user_id + '/' + stage + '/' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            event = EventSet.create_event(line[:-1])
                            if event is not None:
                                event_set.add_event(event)

                    if len(event_set) > 0:
                        event_sets.append(event_set)
                        emotions.append(emotion)
                        segments.append(segment)
                        dates.append(date)
                    else:
                        DebugHelper.warn('empty event set, file: {}{}/{}/{}'.format(file_path, user_id, stage, file))

                if stage == ExperimentStage.TRAIN:
                    FileReader._user_train_event_set[int(user_id)] = event_sets
                    FileReader._user_train_data_emo[int(user_id)] = emotions
                    FileReader._user_train_data_seg[int(user_id)] = segments
                    FileReader._user_train_data_date[int(user_id)] = dates
                elif stage == ExperimentStage.TEST:
                    FileReader._user_test_event_set[int(user_id)] = event_sets
                    FileReader._user_test_data_emo[int(user_id)] = emotions
                    FileReader._user_test_data_seg[int(user_id)] = segments
                    FileReader._user_test_data_date[int(user_id)] = dates

    @staticmethod
    def _read_user_event_seq():
        FileReader.clear_event_seq()
        FileReader._clear_esd()

        file_path: str = Config.event_seq_path
        user_ids = os.listdir(file_path)
        for user_id in user_ids:
            stages = os.listdir(file_path + user_id)
            for stage in stages:
                event_sequences = list()
                emotions = list()
                segments = list()
                dates = list()

                files = os.listdir(file_path + user_id + '/' + stage)
                for file in files:
                    date, segment, emotion, _ = FileReader._parse_file(file)

                    event_sequence = list()
                    with open(file_path + user_id + '/' + stage + '/' + file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            poi = POISet.create_poi(line[:-1])
                            if poi is not None:
                                event_sequence.append(poi)

                    if len(event_sequence) > 0:
                        event_sequences.append(event_sequence)
                        emotions.append(emotion)
                        segments.append(segment)
                        dates.append(date)
                    else:
                        DebugHelper.warn('empty event sequence, file: {}{}/{}/{}'.format(file_path, user_id, stage, file))

                if stage == ExperimentStage.TRAIN:
                    FileReader._user_train_event_seq[int(user_id)] = event_sequences
                    FileReader._user_train_data_emo[int(user_id)] = emotions
                    FileReader._user_train_data_seg[int(user_id)] = segments
                    FileReader._user_train_data_date[int(user_id)] = dates
                elif stage == ExperimentStage.TEST:
                    FileReader._user_test_event_seq[int(user_id)] = event_sequences
                    FileReader._user_test_data_emo[int(user_id)] = emotions
                    FileReader._user_test_data_seg[int(user_id)] = segments
                    FileReader._user_test_data_date[int(user_id)] = dates

    @staticmethod
    def _parse_file(file: str):
        file_name = os.path.splitext(file)[0]
        file_info = file_name.split('-')

        date = '{}-{}-{}'.format(file_info[0], file_info[1], file_info[2])
        segment = int(file_info[3])
        emotion = int(file_info[4])
        is_read = file_info[-1] == 'read'

        return date, segment, emotion, is_read

    @staticmethod
    def _clear_esd():
        FileReader._user_train_data_emo.clear()
        FileReader._user_test_data_emo.clear()
        FileReader._user_train_data_seg.clear()
        FileReader._user_test_data_seg.clear()
        FileReader._user_train_data_date.clear()
        FileReader._user_test_data_date.clear()

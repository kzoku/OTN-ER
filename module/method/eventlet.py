import sys

from module.config.config import Constant, Config
from module.helper.filewriter import FileWriter
from module.helper.helper import DebugHelper
from module.method.event import EventSet
from module.method.poi import POISet
from module.method.stop import StopSet
from module.method.trajectory import TrajectorySet


class Eventlet:
    _TIMER_ID = next(DebugHelper.counter())

    @staticmethod
    def get_sequence(es_set: [StopSet, EventSet], segment: int):
        assert 1 <= segment <= 3

        length = int(Constant.CYCLE / Config.block_size)
        if isinstance(es_set, EventSet):
            sequence = [POISet.get_invalid_poi()] * length
        elif isinstance(es_set, StopSet):
            sequence = [POISet.get_invalid_poi().get_coord()] * length
        else:
            DebugHelper.error('error type! type: {}'.format(type(es_set)))
            sys.exit(1)

        for es in es_set:
            for i in Eventlet._get_range(es.get_start_time(), es.get_end_time(), segment):
                if isinstance(es_set, EventSet):
                    sequence[i] = es.get_poi()
                else:
                    sequence[i] = es.get_center(False)

        return sequence

    @staticmethod
    def get_stop_set(user_gps_data: dict, user_dates: dict, user_segments: dict, user_emotions: dict, stage: str):
        DebugHelper.start(Eventlet._TIMER_ID, 'building [{}] stop set (start)'.format(stage), Config.eventlet_log)

        for user_id, point_set in user_gps_data.items():
            for i, (points, date, segment, emotion) in enumerate(zip(point_set, user_dates[user_id], user_segments[user_id], user_emotions[user_id])):
                DebugHelper.info('current user: {}, trajectory: {}/{}'.format(user_id, i + 1, len(point_set)))
                traj_set = TrajectorySet()
                traj_set.extract_trajectory(points)
                DebugHelper.info('extract finish, {} trajectories extracted'.format(len(traj_set)))
                stop_set = StopSet.extract_stops(traj_set)
                DebugHelper.info('extract finish, {} stops extracted'.format(len(stop_set)))
                FileWriter.record_stop_set(stop_set, user_id, stage, date, segment, emotion)

        DebugHelper.end(Eventlet._TIMER_ID, 'building [{}] stop set (complete)'.format(stage), Config.eventlet_log)

    @staticmethod
    def get_stop_seq(user_stop_sets: dict, user_dates: dict, user_segments: dict, user_emotions: dict, stage: str):
        for user_id, stop_sets in user_stop_sets.items():
            for stop_set, date, segment, emotion in zip(stop_sets, user_dates[user_id], user_segments[user_id], user_emotions[user_id]):
                stop_sequence = Eventlet.get_sequence(stop_set, segment)
                FileWriter.record_stop_seq(stop_sequence, user_id, stage, date, segment, emotion)

    @staticmethod
    def get_event_set(user_stop_sets: dict, user_dates: dict, user_segments: dict, user_emotions: dict, stage: str):
        DebugHelper.start(Eventlet._TIMER_ID, 'building [{}] event set (start)'.format(stage), Config.eventlet_log)

        unknown_coord = list()
        user_event_sets = dict()
        for user_id, stop_sets in user_stop_sets.items():
            user_event_sets[user_id] = list()
            for stop_set in stop_sets:
                event_set = EventSet(stop_set)
                unknown_coord += event_set.find_poi_try()
                user_event_sets[user_id].append(event_set)

        if len(unknown_coord) > 0:
            POISet.create_poi_set(unknown_coord)
            DebugHelper.warn('create POI finish for [{}] unknown stops'.format(len(unknown_coord)))

        for user_id, event_sets in user_event_sets.items():
            for event_set, date, segment, emotion in zip(event_sets, user_dates[user_id], user_segments[user_id], user_emotions[user_id]):
                event_set.find_poi_final()
                FileWriter.record_unknown_event(event_set.get_unknown_event(), False)
                FileWriter.record_event_set(event_set, user_id, stage, date, segment, emotion)

        DebugHelper.end(Eventlet._TIMER_ID, 'building [{}] event set (complete)'.format(stage), Config.eventlet_log)

    @staticmethod
    def get_event_seq(user_event_sets: dict, user_dates: dict, user_segments: dict, user_emotions: dict, stage: str):
        for user_id, event_sets in user_event_sets.items():
            for event_set, date, segment, emotion in zip(event_sets, user_dates[user_id], user_segments[user_id], user_emotions[user_id]):
                event_sequence = Eventlet.get_sequence(event_set, segment)
                FileWriter.record_event_seq(event_sequence, user_id, stage, date, segment, emotion)

    @staticmethod
    def _get_range(start_time: int, end_time: int, segment: int):
        start_time %= 86400
        end_time %= 86400

        length_1 = int(Constant.START / Config.block_size)
        length_2 = int(Constant.CYCLE / Config.block_size)

        start = int(start_time / Config.block_size + 0.5) - length_1 - length_2 * (segment - 1)
        end = int(end_time / Config.block_size - 0.5) - length_1 - length_2 * (segment - 1)

        return range(start, end)

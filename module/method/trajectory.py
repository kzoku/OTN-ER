import sys

from module.config.config import Config, Constant
from module.helper.helper import DebugHelper, GeoHelper, TimeHelper, IterableBase


class Point:
    def __init__(self, coord: tuple[float, float], timestamp: int):
        self._coord = coord
        self._timestamp = timestamp

    def __gt__(self, other: 'Point'):
        return self._timestamp > other._timestamp

    def __repr__(self):
        return Constant.SEPARATOR.join(map(str, [self._coord, self._timestamp]))

    def to_string(self):
        return Constant.SEPARATOR.join(map(str, [self._coord, TimeHelper.int2utc(self._timestamp)]))

    def get_coord(self):
        return self._coord

    def get_timestamp(self):
        return self._timestamp


class TrajectorySet(IterableBase):
    _GOOD = 0
    _A_ERROR = 1
    _B1_ERROR = 2
    _B2_ERROR = 3
    _B3_ERROR = 4
    _B4_ERROR = 5

    _MIN_COORD = 2
    _MAX_SPEED = 1000
    _TIME_GAP_1 = 900
    _TIME_GAP_2 = 3600
    _TIME_GAP_3 = 14400
    _TIME_GAP_4 = 86400

    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self._data)

    def extract_trajectory(self, points: list):
        trajectory = list()
        state_code = TrajectorySet._GOOD

        for cur_point in points:
            if len(trajectory) == 0:
                trajectory.append(cur_point)
                continue

            distance = 0
            last_point = trajectory[-1]

            time_gap = cur_point.get_timestamp() - last_point.get_timestamp()
            if time_gap <= 0:
                DebugHelper.error('time gap exception, point = [{}, {}]'.format(cur_point, last_point))
                sys.exit(1)
            elif time_gap >= TrajectorySet._TIME_GAP_4:
                state_code = TrajectorySet._B4_ERROR
            elif time_gap >= TrajectorySet._TIME_GAP_3:
                state_code = TrajectorySet._B3_ERROR
            elif time_gap >= TrajectorySet._TIME_GAP_2:
                state_code = TrajectorySet._B2_ERROR
            elif time_gap >= TrajectorySet._TIME_GAP_1:
                distance = GeoHelper.get_distance(cur_point.get_coord(), last_point.get_coord())
                if distance > Config.dist_th:
                    state_code = TrajectorySet._B1_ERROR

            if state_code == TrajectorySet._GOOD:
                distance = distance if distance > 0 else GeoHelper.get_distance(cur_point.get_coord(), last_point.get_coord())
                if distance / time_gap > TrajectorySet._MAX_SPEED:
                    state_code = TrajectorySet._A_ERROR

            if state_code != TrajectorySet._GOOD:
                state_code = TrajectorySet._GOOD
                if len(trajectory) >= TrajectorySet._MIN_COORD:
                    self._data.append(trajectory)

                trajectory = list()

            trajectory.append(cur_point)

        if len(trajectory) >= TrajectorySet._MIN_COORD:
            self._data.append(trajectory)

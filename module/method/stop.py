from typing import Union

from module.config.config import Config, Constant
from module.helper.helper import GeoHelper, ContainerHelper, TimeHelper, IterableBase
from module.method.trajectory import TrajectorySet


class Stop:
    def __new__(cls, points: Union[list, None] = None, center: Union[tuple[float, float], None] = None, start_time: Union[int, None] = None, end_time: Union[int, None] = None):
        if (points is None or len(points) < 2) and (center is None or start_time is None or end_time is None):
            return None

        return super(Stop, cls).__new__(cls)

    def __init__(self, points: Union[list, None] = None, center: Union[tuple[float, float], None] = None, start_time: Union[int, None] = None, end_time: Union[int, None] = None):
        self._points = points
        if points is not None and len(points) > 1:
            self._center = self.get_center(True)
            self._start_time = self._points[0].get_timestamp()
            self._end_time = self._points[-1].get_timestamp()
        else:
            self._center = center
            self._start_time = start_time
            self._end_time = end_time

    def __add__(self, other: 'Stop'):
        if not isinstance(other, Stop):
            return None

        other_points = other.get_points()
        if self._points is None or other_points is None:
            max_start_time = max(self._start_time, other.get_start_time())
            min_end_time = min(self._end_time, other.get_end_time())

            if max_start_time - min_end_time < 0:
                return None

            if max_start_time - min_end_time > Config.time_th:
                return None

            center = ContainerHelper.get_average([self._center, other.get_center(False)], 0)
            min_start_time = min(self._start_time, other.get_start_time())
            max_end_time = max(self._end_time, other.get_end_time())
            return Stop(center=center, start_time=min_start_time, end_time=max_end_time)
        else:
            points = self._points + other_points
            points.sort()
            return Stop(points)

    def __repr__(self):
        return Constant.SEPARATOR.join(map(str, [self._center, self._start_time, self._end_time]))

    def to_string(self):
        return Constant.SEPARATOR.join(map(str, [self._center, TimeHelper.int2utc(self._start_time), TimeHelper.int2utc(self._end_time)]))

    def to_vector(self):
        return [*self._center, self._start_time, self._end_time]

    def get_points(self):
        return self._points

    def get_center(self, calculate: bool):
        if calculate:
            self._center = ContainerHelper.get_average([point.get_coord() for point in self._points], 0)

        return self._center

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_duration(self):
        return self._end_time - self._start_time

    def copy(self):
        if self._points is None:
            return Stop(center=self._center, start_time=self._start_time, end_time=self._end_time)
        else:
            return Stop(points=self._points)


class StopSet(IterableBase):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return '\n'.join(map(str, self._data)) + '\n'

    def __len__(self):
        return len(self._data)

    def add_stop(self, stop: Stop):
        self._data.append(stop)

    def to_vector(self):
        return [stop.to_vector() for stop in self._data]

    def to_string(self):
        string = ''
        for i, stop in enumerate(self._data):
            string += 'The {}th stop: {}, duration: {}\n'.format(i + 1, stop.to_string(), TimeHelper.second2str(stop.get_duration()))

        return string

    @staticmethod
    def extract_stops(traj_set: TrajectorySet):
        stop_set = StopSet()
        for points in traj_set:
            coords = [point.get_coord() for point in points]

            start = 0
            end = start + 1
            while start < len(points) and end < len(points):
                while end < len(points) and not StopSet._test_tth(points[start:end + 1]):
                    end += 1

                if end == len(points) or not StopSet._test_tth(points[start:end + 1]):
                    break

                if not StopSet._test_dth(coords[start:end + 1]):
                    start += 1
                    end = max(end, start + 1)
                    continue

                while end + 1 < len(points):
                    if coords[end] == coords[end + 1] or StopSet._quick_test_dth(coords[start:end + 2]) or StopSet._test_dth(coords[start:end + 2]):
                        end += 1
                        continue
                    else:
                        break

                stop_set.add_stop(Stop(points=points[start:end + 1]))

                start = end + 1
                end = start + 1

        return stop_set

    @staticmethod
    def create_stop(stop_str: str):
        if len(stop_str) == 0:
            return None

        info = stop_str.split(Constant.SEPARATOR)
        center = GeoHelper.parse_coord(info[0] + ',' + info[1])
        return Stop(center=center, start_time=int(info[2]), end_time=int(info[3]))

    @staticmethod
    def _test_tth(points: list):
        return points[-1].get_timestamp() - points[0].get_timestamp() >= Config.time_th

    @staticmethod
    def _test_dth(coords: list):
        center = ContainerHelper.get_average(coords, 0)
        for coord in coords:
            if GeoHelper.get_distance(coord, center) > Config.dist_th:
                return False

        return True

    @staticmethod
    def _quick_test_dth(coords: list):
        center = ContainerHelper.get_average(coords[0:-1], 0)
        return GeoHelper.get_distance(coords[-1], center) <= Config.dist_th

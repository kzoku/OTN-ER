import sys

import numpy as np
from sklearn import cluster

from module.config.config import Config, Constant
from module.helper.helper import DebugHelper, GeoHelper, TimeHelper, OneTimeAssign


class POI(OneTimeAssign):
    def __init__(self, name: str, coord: tuple[float, float], ptype: int):
        self._name = name
        self._coord = coord
        self._ptype = ptype

    def __hash__(self):
        return hash((self._name, self._coord, self._ptype))

    def __repr__(self):
        return Constant.SEPARATOR.join(map(str, [self._name, self._coord, self._ptype]))

    def __eq__(self, other: 'POI'):
        if not isinstance(other, POI):
            return False

        return self._name == other.get_name() and self._ptype == other.get_ptype() and GeoHelper.get_distance(self._coord, other.get_coord()) <= Constant.BEST_DIST_TH

    def to_vector(self):
        return [self._ptype, *self._coord]

    def get_name(self):
        return self._name

    def get_coord(self):
        return self._coord

    def get_ptype(self):
        return self._ptype

    def is_valid(self):
        return self._name != POISet.INVALID_NAME and self._coord != POISet.INVALID_COORD and self._ptype != POISet.INVALID_PTYPE


class POISet:
    INVALID_NAME = 'Invalid POI'
    INVALID_COORD = (0.0, 0.0)
    INVALID_PTYPE = -1

    UNKNOWN_PTYPE = 0
    EATING_PTYPE = 1
    RESTING_PTYPE = 2
    STUDYING_PTYPE = 3
    WORKING_PTYPE = 4
    SPORTS_PTYPE = 5
    ADMIN_PTYPE = 6
    EMERGENCY_PTYPE = 7
    ACCESS_PTYPE = 8
    NECESSARY_PTYPE = 9
    ENTERTAINMENT_PTYPE = 10

    _RECOMMEND = {
        '0:00:00-7:30:00': RESTING_PTYPE,
        '11:30:00-13:30:00': EATING_PTYPE,
        '17:00:00-19:00:00': EATING_PTYPE,
        '23:30:00-23:59:59': RESTING_PTYPE,
    }

    _base_data = list()
    _create_data = list()

    @staticmethod
    def set_base_poi_set(data: list):
        POISet._base_data = data

    @staticmethod
    def get_base_poi_set():
        return POISet._base_data

    @staticmethod
    def get_create_poi_set():
        return POISet._create_data

    @staticmethod
    def get_invalid_poi():
        return POI(POISet.INVALID_NAME, POISet.INVALID_COORD, POISet.INVALID_PTYPE)

    @staticmethod
    def get_recommend_type(t_1: int, t_2: int):
        t_1 = TimeHelper.seconds_of_day(t_1)
        t_2 = TimeHelper.seconds_of_day(t_2)
        for time_range, rtype in POISet._RECOMMEND.items():
            tr = time_range.split('-')
            tr_1 = POISet._time2int(tr[0])
            tr_2 = POISet._time2int(tr[1])
            if tr_1 <= t_1 and t_2 <= tr_2:
                return rtype

        return POISet.INVALID_PTYPE

    @staticmethod
    def create_poi_set(coords: list):
        if len(coords) == 0:
            return

        coords = np.array(coords)

        try_again = True
        n_clusters = 1
        clustering = None
        while try_again:
            try_again = False
            clustering = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            clustering.fit(coords)

            for n_class, centroid in enumerate(clustering.cluster_centers_):
                for coord in coords[clustering.labels_ == n_class]:
                    if GeoHelper.get_distance(coord, centroid) > Config.dist_th:
                        try_again = True
                        n_clusters += 1
                        break

                if try_again:
                    break

        for n_class, centroid in enumerate(clustering.cluster_centers_):
            count = len(coords[clustering.labels_ == n_class])
            POISet._create_data.append(POI('Cluster: [{}]'.format(count), tuple(centroid), POISet.UNKNOWN_PTYPE))

    @staticmethod
    def create_poi(poi_str: str):
        if len(poi_str) == 0:
            return None

        info = poi_str.split(Constant.SEPARATOR)
        coord = GeoHelper.parse_coord(info[1] + ',' + info[2])
        if coord != POISet.INVALID_COORD:
            DebugHelper.error('error poi str, poi_str: {}'.format(poi_str))
            sys.exit(1)

        return POI(name=info[0], coord=coord, ptype=int(info[3]))

    @staticmethod
    def _time2int(time_str: str):
        hour, minute, second = map(int, time_str.split(':'))
        return hour * 3600 + minute * 60 + second

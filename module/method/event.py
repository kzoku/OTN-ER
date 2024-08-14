import sys
from collections import Counter
from typing import Union

from module.config.config import Config, Constant
from module.helper.helper import DebugHelper, GeoHelper, TimeHelper, IterableBase
from module.method.poi import POI, POISet
from module.method.stop import Stop, StopSet


# Event = Stop + POI
class Event:
    def __new__(cls, stop: Union['Stop', None] = None, poi: Union['POI', None] = None):
        if stop is None:
            return None

        return super(Event, cls).__new__(cls)

    def __init__(self, stop: Union['Stop', None] = None, poi: Union['POI', None] = None):
        self._stop = stop
        self._poi = poi
        self._near_poi = list() if poi is None else [poi]

    def __repr__(self):
        return Constant.SEPARATOR.join(map(str, [self._stop, self._poi]))

    def __add__(self, other: 'Event'):
        if not isinstance(other, Event):
            return None

        other_poi = other.get_poi()
        if self._poi is None or other_poi is None or self._poi != other_poi:
            return None

        new_stop = self._stop + other.get_stop()
        if new_stop is None:
            return None

        return Event(stop=new_stop, poi=self._poi)

    def to_string(self):
        return Constant.SEPARATOR.join(map(str, [self._stop.to_string(), self._poi]))

    def to_vector(self):
        stop_vec = self._stop.to_vector()
        poi_vec = self._poi.to_vector()
        return [*stop_vec, *poi_vec]

    def get_stop(self):
        return self._stop

    def get_poi(self):
        return self._poi

    def get_start_time(self):
        return self._stop.get_start_time()

    def get_end_time(self):
        return self._stop.get_end_time()

    def get_duration(self):
        return self._stop.get_duration()

    def get_near_poi(self):
        return self._near_poi

    def is_unknown_event(self):
        return self._poi is None or not self._poi.is_valid() or self._poi.get_ptype() == POISet.UNKNOWN_PTYPE

    def copy(self):
        return Event(stop=self._stop.copy(), poi=self._poi)

    def find_near_poi(self, poi_set: list):
        if len(self._near_poi) > 0:
            return True

        center = self._stop.get_center(False)
        for poi in poi_set:
            if GeoHelper.get_distance(center, poi.get_coord()) <= Config.dist_th:
                self._near_poi.append(poi)

        self._near_poi = list(set(self._near_poi))

        return len(self._near_poi) > 0

    def find_best_poi(self, prev_event: Union['Event', None] = None):
        if len(self._near_poi) == 0:
            DebugHelper.error('no near POI, event: {}'.format(str(self)))
            sys.exit(1)

        if len(self._near_poi) == 1:
            self._poi = self._near_poi[0]
            return

        find_flag = False

        if not find_flag:
            ptype = [poi.get_ptype() for poi in self._near_poi]
            most_com = Counter(ptype).most_common(1)
            if most_com[0][1] > 1:
                mtype = most_com[0][0]
                candidate_poi = list()
                for poi in self._near_poi:
                    if poi.get_ptype() == mtype:
                        candidate_poi.append(poi)

                find_flag = self._find_nearest_poi(candidate_poi)

        if not find_flag:
            rtype = POISet.get_recommend_type(self._stop.get_start_time(), self._stop.get_end_time())
            if rtype != POISet.INVALID_PTYPE:
                candidate_poi = list()
                for poi in self._near_poi:
                    if poi.get_ptype() == rtype:
                        candidate_poi.append(poi)

                find_flag = self._find_nearest_poi(candidate_poi)

        if not find_flag:
            if prev_event is not None:
                ptype = prev_event.get_poi().get_ptype()
                candidate_poi = list()
                for poi in self._near_poi:
                    if poi.get_ptype() == ptype:
                        candidate_poi.append(poi)

                find_flag = self._find_nearest_poi(candidate_poi)

        if not find_flag:
            self._find_nearest_poi(self._near_poi)

    def _find_nearest_poi(self, candidate_poi: list):
        if len(candidate_poi) == 0:
            return False

        center = self._stop.get_center(False)
        min_dist = sys.maxsize
        for poi in candidate_poi:
            dist = GeoHelper.get_distance(center, poi.get_coord())
            if dist < min_dist:
                min_dist = dist
                self._poi = poi

        return True


class EventSet(IterableBase):
    def __new__(cls, stop_set: Union[StopSet, None] = None):
        return super(EventSet, cls).__new__(cls)

    def __init__(self, stop_set: Union[StopSet, None] = None):
        super().__init__()

        if stop_set is not None:
            for stop in stop_set:
                self._data.append(Event(stop=stop, poi=None))

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return '\n'.join(map(str, self._data)) + '\n'

    def __add__(self, other: 'EventSet'):
        self._index = 0
        for event in other:
            self._data.append(event.copy())

        return self

    def to_vector(self):
        events = list()
        for event in self._data:
            events.append(event.to_vector())

        return events

    def to_string(self):
        string = ''
        for i, event in enumerate(self._data):
            string += 'The {}th event: {}, duration: {}\n'.format(i + 1, event.to_string(), TimeHelper.second2str(event.get_duration()))

        return string

    def add_event(self, event: Event):
        self._data.append(event)

    def get_unknown_event(self):
        unknown = list()
        for event in self._data:
            if event.is_unknown_event():
                unknown.append(event)

        return unknown

    def find_poi_try(self):
        poi_set = POISet.get_base_poi_set()
        un_find = self._find_near_poi(poi_set)
        return un_find

    def find_poi_final(self):
        poi_set = POISet.get_create_poi_set()
        un_find = self._find_near_poi(poi_set)
        if len(un_find) > 0:
            DebugHelper.error('{} stops can not find POI'.format(len(un_find)))
            sys.exit(1)

        self._find_best_poi()
        self._merge_event()

    def _find_near_poi(self, poi_set: list):
        un_find = list()
        for event in self._data:
            if not event.find_near_poi(poi_set):
                un_find.append(event.get_stop().get_center(False))

        return un_find

    def _find_best_poi(self):
        partition = 0
        find_flag = False

        for i, event in enumerate(self._data):
            if len(event.get_near_poi()) == 1:
                partition = i
                find_flag = True
                event.find_best_poi(None)
                break

        if find_flag and partition > 0:
            prev_event = self._data[partition]
            sub_events = self._data[:partition]
            for event in reversed(sub_events):
                event.find_best_poi(prev_event)
                prev_event = event

        if find_flag and partition < len(self._data) - 1:
            prev_event = self._data[partition]
            sub_events = self._data[partition + 1:]
            for event in sub_events:
                event.find_best_poi(prev_event)
                prev_event = event

        if not find_flag:
            for event in self._data:
                event.find_best_poi(None)

    def _merge_event(self):
        if len(self._data) < 2:
            return

        new_events = list()
        last_event = self._data[0]
        for cur_event in self._data[1:]:
            new_event = cur_event + last_event
            if new_event is None:
                new_events.append(last_event)
                last_event = cur_event
            else:
                last_event = new_event

        new_events.append(last_event)
        self._data = new_events

    @staticmethod
    def create_event(event_str: str):
        if len(event_str) == 0:
            return None

        info = event_str.split(Constant.SEPARATOR)
        stop_str = Constant.SEPARATOR.join(info[:4])
        poi_str = Constant.SEPARATOR.join(info[4:])
        return Event(stop=StopSet.create_stop(stop_str), poi=POISet.create_poi(poi_str))

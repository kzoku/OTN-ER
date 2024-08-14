class EmotionType:
    NEGATIVE = 1
    NEUTRAL = 2
    POSITIVE = 3


class SegmentType:
    MORNING = 1
    AFTERNOON = 2
    EVENING = 3


class ExperimentType:
    EFFECTIVENESS = 'effectiveness'
    TIME_TH = 'time_th'
    BLOCK_SIZE = 'block_size'
    DATA_TYPE = 'data_type'


class ExperimentStage:
    TRAIN = 'train'
    TEST = 'test'


class Constant:
    SEPARATOR = ', '

    DATA_SET_1 = 'XDU_1'
    DATA_SET_2 = 'XDU_2'
    DATA_SETS = [DATA_SET_1, DATA_SET_2]

    DATA_SSET = 'stop_set'
    DATA_SSEQ = 'stop_seq'
    DATA_ESET = 'event_set'
    DATA_ESEQ = 'event_seq'
    DATA_TYPES = [DATA_SSET, DATA_SSEQ, DATA_ESET, DATA_ESEQ]

    POISET_PATH = './poiset/'

    KNN = 'KNN'
    KMeans = 'K-Means'
    ROCKET = 'ROCKET'
    KShape = 'K-Shape'
    MLP = 'MLP'
    RNN = 'RNN'
    GRU = 'GRU'
    LSTM = 'LSTM'
    Transformer = 'Transformer'

    BL_METHODS = [KNN, KMeans, MLP]
    TS_METHODS = [RNN, GRU, LSTM, Transformer]
    ST_METHODS = [KShape]

    CYCLE = 21600
    START = 21600

    BEST_DATA_TYPE = DATA_ESEQ
    BEST_TIME_TH = 900
    BEST_DIST_TH = 200
    BEST_BLOCK_SIZE = 600

    TIME_THS = [600, 900, 1200, 1500, 1800]
    BLOCK_SIZES = [300, 600, 900]


class Config:
    data_set = ''
    time_th = 0
    dist_th = 200
    block_size = 0

    exp_log_path = ''
    eventlet_log = ''
    npy_data_path = ''
    gps_data_path = ''
    stop_set_path = ''
    stop_seq_path = ''
    event_set_path = ''
    event_seq_path = ''

    @staticmethod
    def set_config(data_set: str, time_th: int, block_size: int):
        Config.data_set = data_set
        Config.time_th = time_th
        Config.block_size = block_size

        Config.eventlet_log = './output/{}/log.txt'.format(data_set)
        Config.npy_data_path = '../_ShareInput/Emotion/{}_npy/'.format(data_set)
        Config.gps_data_path = '../_ShareInput/Emotion/{}_GPSData/'.format(data_set)
        Config.stop_set_path = '../_ShareInput/Emotion/{}_StopSet_{}/'.format(data_set, Config.time_th)
        Config.stop_seq_path = '../_ShareInput/Emotion/{}_StopSeq_{}_{}/'.format(data_set, Config.time_th, Config.block_size)
        Config.event_set_path = '../_ShareInput/Emotion/{}_EventSet_{}/'.format(data_set, Config.time_th)
        Config.event_seq_path = '../_ShareInput/Emotion/{}_EventSeq_{}_{}/'.format(data_set, Config.time_th, Config.block_size)

    @staticmethod
    def set_log_path(data_set: str, exp_type: str):
        Config.exp_log_path = './output/{}/log/{}/'.format(data_set, exp_type)

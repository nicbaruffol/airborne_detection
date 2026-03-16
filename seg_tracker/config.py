import os

# Look for 'FAST_DATA_DIR'. If it's not set, default to your scratch folder.
DATA_DIR = os.environ.get('FAST_DATA_DIR', '/cluster/scratch/nbaruffol/airborne_dataset_new')
SRC_DATA_DIR = os.environ.get('FAST_DATA_DIR', '/cluster/scratch/nbaruffol/airborne_dataset_new')

IMG_FORMAT = 'jpg'

UPPER_BOUND_MIN_DIST = 330
UPPER_BOUND_MAX_DIST = 700
UPPER_BOUND_MAX_DIST_SELECTED_TRAIN = 1000
MAX_PREDICT_DISTANCE = 2000

OFFSET_SCALE = 256.0

MIN_OBJECT_AREA = 100   
IS_MATCH_MIN_IOU_THRESH = 0.2
IS_NO_MATCH_MAX_IOU_THRESH = 0.02
MIN_SECS = 3.0

CLASSES = [
    'None',
    'Airborne',
    'Airplane',
    'Bird',
    'Drone',
    'Flock',
    'Helicopter'
]

NB_CLASSES = len(CLASSES)


TRANSFORM_MODEL = '030_tr_tsn_rn34_w3_crop_borders'
TRANSFORM_MODEL_EPOCH = 255
TRANSFORM_MODEL_FOLD = 0

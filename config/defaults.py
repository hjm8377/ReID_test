from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = "0"
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.NAME = ''
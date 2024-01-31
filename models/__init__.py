# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------

'''build models'''

# from .tadtr import build
# from .DABDETR import build
from .DABDETR_02 import build
# from .dino import build

def build_model(args):
    return build(args)

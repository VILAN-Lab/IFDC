from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .AttModel import *

def setup(opt):
    # Top-down attention model
    if opt.caption_model in ['topdown', 'updown']:
        model = UpDownModel(opt)

    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model

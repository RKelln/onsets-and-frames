import os

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

from onsets_and_frames import *

ex = Experiment('convert_transcriber')


@ex.config
def config():
    modeldir = 'models/uni/'
    resume_iteration = 1000000

    model_complexity = 48

    bidirectional = False


@ex.automain
def convert(modeldir, resume_iteration, model_complexity):
    print_config(ex.current_run)

    existing = os.path.join(modeldir, f'model-{resume_iteration}.pt')
    state_dict = os.path.join(modeldir, f'model-{resume_iteration}.state_dict')
    realtime = os.path.join(modeldir, f'model-rt-{resume_iteration}-pt1.6.pt')

    # load existing model and save its state dict
    #model = torch.load(existing)
    #torch.save(model.state_dict(), state_dict)

    # create new model and set its state dict
    model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity, False)
    model.load_state_dict(torch.load(state_dict))
    
    # to deal with torch.nn.modules.module.ModuleAttributeError: 'BatchNorm2d' object has no attribute '_non_persistent_buffers_set'
    # https://github.com/ultralytics/yolov5/issues/58
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

    summary(model)

    torch.save(model, realtime)

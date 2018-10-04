from __future__ import print_function

from collections import defaultdict
from pathlib import Path

import dill as pickle
import scipy.io.wavfile as wavfile
from python_speech_features import *
from tqdm import tqdm

"""Purpouse of this script is to do the required preprocessing and feature extraction from the dataset."""


def save_obj(obj, name):
    """Save object to a file, using dill package, as pickle is not suitable for files over 2GB."""
    with open(SAVE_PATH / name, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)


def load_obj(name):
    """Load object from a file using dill."""
    with open(SAVE_PATH / name, 'rb') as f:
        return pickle.load(f)


DATA_PATH = Path("../../dataset")
SAVE_PATH = Path("/opt/project")
DATA_FILE = "data.pickle"
LAB_IDX_FILE = "to_idx.pickle"
IDX_LAB_FILE = "to_lab.pickle"
CUTS = [0]

i = 0
while CUTS[i] < 200000:
    CUTS.append(CUTS[i] + 483)
    i += 1

labels = []
data = defaultdict(list)

for dirs in tqdm(DATA_PATH.iterdir()):
    labels.append(str(dirs.name))

    for file in dirs.iterdir():
        sample_rate, signal = wavfile.read(file)

        signal = signal[signal != 0]

        mfcc_feat = mfcc(signal, sample_rate, winlen=0.01, winstep=0.0005)

        mfcc_feats = []
        i = 1
        while CUTS[i] <= len(mfcc_feat):
            mfcc_feats.append(mfcc_feat[CUTS[i - 1]:CUTS[i]])
            i += 1

        data[dirs.name].append(mfcc_feats)

label_to_idx = {lab: i for i, lab in enumerate(labels)}
idx_to_label = {i: lab for i, lab in enumerate(labels)}

save_obj(label_to_idx, LAB_IDX_FILE)
save_obj(idx_to_label, IDX_LAB_FILE)
save_obj(data, DATA_FILE)

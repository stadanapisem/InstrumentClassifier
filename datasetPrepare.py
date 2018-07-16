from __future__ import print_function

from collections import defaultdict
from pathlib import Path

import dill as pickle
import scipy.io.wavfile as wavfile
from python_speech_features import *
from tqdm import tqdm


def save_obj(obj, name):
    with open(SAVE_PATH / name, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)


def load_obj(name):
    with open(SAVE_PATH / name, 'rb') as f:
        return pickle.load(f)


DATA_PATH = Path("../../dataset")
SAVE_PATH = Path("/opt/project")
DATA_FILE = "data_small.pickle"
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
    if str(dirs.name) == 'flute':
        break
    for file in dirs.iterdir():
        sample_rate, signal = wavfile.read(file)

        """search = re.search('\A(.*)_(\w+\d)_(\d+)_.*', file.name)
        try:
            file_length = int(search.group(3))
        except AttributeError:
            file_length = 
        if file_length == 5:
            file_length = 0.5
        elif file_length == 25:
            file_length = 0.25"""

        # signal = signal[0:int(sample_rate * file_length)]
        # signal = np.trim_zeros(signal)
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

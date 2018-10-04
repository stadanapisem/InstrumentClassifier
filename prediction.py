from InstrumentsDataset import *
import argparse
import scipy.io.wavfile as wavfile
from python_speech_features import *
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)

    args = parser.parse_args()
    print(args.file)

    label_to_idx = load_obj(LAB_IDX_FILE)
    idx_to_label = load_obj(IDX_LAB_FILE)
    print(label_to_idx)

    sample_rate, signal = wavfile.read(args.file)

    CUTS = [0]

    i = 0
    while CUTS[i] < 200000:
        CUTS.append(CUTS[i] + 483)
        i += 1

    model = torch.load("model_best.pytorch")
    model.eval()

    signal = signal[signal != 0]
    mfcc_feat = mfcc(signal, sample_rate, winlen=0.01, winstep=0.0005)

    i = 1
    predictions = np.zeros((1, 2), dtype=float)
    while CUTS[i] <= len(mfcc_feat):
        data = Variable(torch.cuda.DoubleTensor(mfcc_feat[CUTS[i - 1]:CUTS[i]]))
        data.unsqueeze_(0)
        pred = model(data)
        predictions += F.softmax(pred, dim=1).data.cpu().numpy()
        i += 1

    predictions /= (i - 1)
    print(idx_to_label[np.argmax(predictions)])


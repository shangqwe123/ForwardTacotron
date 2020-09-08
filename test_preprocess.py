from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import webrtcvad
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics.pairwise import cosine_similarity
import librosa
from matplotlib import pyplot as plt
import struct
from utils.display import plot_cos_matrix
from utils.dsp import load_wav, melspectrogram, trim_long_silences
from utils.files import pickle_binary
from utils import hparams as hp


def plot_mel(wav, path):
    mel = melspectrogram(wav).astype(np.float32)
    mel = np.flip(mel, axis=0)
    plt.imshow(mel, interpolation='nearest', aspect='auto')
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':

    hp.configure('hparams.py')

    np.set_printoptions(precision=3, suppress=True)

    encoder = VoiceEncoder()

    file = Path('/Users/cschaefe/datasets/VCTK-Corpus/wav48/p225/p225_003.wav')

    wav_orig, sr = librosa.load(file, sr=22050)
    wav = trim_long_silences(wav_orig)

    plot_mel(wav_orig, f'/tmp/preproc/{file.stem}_norm_orig.png')
    plot_mel(wav, f'/tmp/preproc/{file.stem}_norm.png')

    librosa.output.write_wav(f'/tmp/preproc/{file.stem}_norm_orig.wav', wav_orig, sr=sr)
    librosa.output.write_wav(f'/tmp/preproc/{file.stem}_norm.wav', wav, sr=sr)

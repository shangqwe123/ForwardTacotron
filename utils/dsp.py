import math
import torch
import numpy as np
import librosa
#from utils import hparams as hp
from scipy.signal import lfilter

from utils.audio_processing import griffin_lim
from utils.stft import TacotronSTFT

from scipy.io.wavfile import read

# hack this for now
class hp:
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    num_mels = 80
    sample_rate = 22050
    fmin = 0
    fmax = 8000

taco_stft = TacotronSTFT(
    filter_length=hp.n_fft, hop_length=hp.hop_length,
    win_length=hp.win_length, n_mel_channels=hp.num_mels,
    sampling_rate=hp.sample_rate, mel_fmin=hp.fmin, mel_fmax=hp.fmax)


def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def read_wav_np(path):
   sr, wav = read(path)

   if len(wav.shape) == 2:
      wav = wav[:, 0]

   if wav.dtype == np.int16:
      wav = wav / 32768.0
   elif wav.dtype == np.int32:
      wav = wav / 2147483648.0
   elif wav.dtype == np.uint8:
      wav = (wav - 128) / 128.0

   wav = wav.astype(np.float32)

   return sr, wav


def load_wav(path):
    sr, wav = read_wav_np(path)
    assert sr == 22050, f'sampling rate was {sr}'
    return wav


def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)


def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

'''
def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)
'''



def melspectrogram(y):
    y = torch.tensor(y).unsqueeze(0).float()
    return taco_stft.mel_spectrogram(y).squeeze().numpy()

def raw_melspec(y):
    D = stft(y)
    S = linear_to_mel(np.abs(D))
    return S



def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def pre_emphasis(x):
    return lfilter([1, -hp.preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -hp.preemphasis], x)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels: y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    mel = torch.tensor(mel).float()
    mel = taco_stft.spectral_de_normalize(mel).numpy()
    S = librosa.feature.inverse.mel_to_stft(
        mel, power=1, sr=hp.sample_rate,
        n_fft=hp.n_fft, fmin=hp.fmin, fmax=hp.fmax)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hp.hop_length, win_length=hp.win_length)
    return wav

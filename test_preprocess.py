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
from utils.dsp import load_wav, melspectrogram
from utils.files import pickle_binary
from utils import hparams as hp


## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30


## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
int16_max = (2 ** 15) - 1

def trim_long_silences(wav):
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    voice_indices = np.where(audio_mask)[0]
    voice_start, voice_end = voice_indices[0], voice_indices[-1]
    audio_mask[voice_start:voice_end] = binary_dilation(audio_mask[voice_start:voice_end], np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask==True]


def normalize_volume(wav, target_dBFS):
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    return wav * (10 ** (dBFS_change / 20))


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
    wav_orig = normalize_volume(wav_orig, target_dBFS=-30)
    wav = trim_long_silences(wav_orig)

    plot_mel(wav_orig, f'/tmp/preproc/{file.stem}_norm_orig.png')
    plot_mel(wav, f'/tmp/preproc/{file.stem}_norm.png')

    librosa.output.write_wav(f'/tmp/preproc/{file.stem}_norm_orig.wav', wav_orig, sr=sr)
    librosa.output.write_wav(f'/tmp/preproc/{file.stem}_norm.wav', wav, sr=sr)

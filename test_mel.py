import numpy as np
from utils import hparams as hp
import argparse

from utils.audio_processing import griffin_lim
from utils.dsp import load_wav, melspectrogram, reconstruct_waveform
from utils.stft import TacotronSTFT

from scipy.io.wavfile import read

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

if __name__ == '__main__':

   # Parse Arguments
   parser = argparse.ArgumentParser(description='TTS Generator')
   parser.set_defaults(input_text=None)
   parser.set_defaults(weights_path=None)

   args = parser.parse_args()

   hp.configure('hparams.py')  # Load hparams from file

   path = '/Users/cschaefe/datasets/audio_data/Cutted_merged_resampled/02386.wav'

   wav = load_wav(path)

   mel = melspectrogram(wav)

   print(mel.shape)

   wav = reconstruct_waveform(mel)

   import librosa
   librosa.output.write_wav('/tmp/sample.wav', wav, sr=22050)
   print(wav.shape)


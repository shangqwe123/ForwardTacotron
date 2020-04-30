import torch
import torch.nn.functional as F
from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils import hparams as hp
from utils.distribution import STFTLoss, TacotronSTFT
from utils.files import unpickle_binary
from utils.text.symbols import phonemes
from utils.paths import Paths
import argparse
from utils.text import text_to_sequence, clean_text, sequence_to_text
from utils.display import simple_table, plot_mel
from utils.dsp import reconstruct_waveform, save_wav, load_wav, raw_melspec
import numpy as np

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Train Tacotron TTS')
   parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
   parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
   parser.add_argument('--force_cpu', '-c', action='store_true',
                       help='Forces CPU-only training, even when in CUDA capable environment')
   parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                       help='The file to use for the hyperparameters')
   args = parser.parse_args()

   hp.configure(args.hp_file)  # Load hparams from file

   wav_1 = load_wav('/Users/cschaefe/Downloads/target.wav')
   wav_2 = load_wav('/Users/cschaefe/Downloads/individualAudio_1.wav')
   wav_3 = load_wav('/Users/cschaefe/Downloads/individualAudio.wav')

   mel_1 = raw_melspec(wav_1)
   mel_2 = raw_melspec(wav_2)
   mel_3 = raw_melspec(wav_3)
   mel_1 = torch.tensor(mel_1).unsqueeze(0)
   mel_2 = torch.tensor(mel_2).unsqueeze(0)
   mel_3 = torch.tensor(mel_3).unsqueeze(0)

   loss_1 = F.l1_loss(mel_2, mel_1)
   loss_2 = F.l1_loss(mel_3, mel_1)

   mel_1 = plot_mel(mel_1.squeeze().numpy())
   mel_1.savefig('/tmp/mel_1.png')
   mel_2 = plot_mel(mel_2.squeeze().numpy())
   mel_2.savefig('/tmp/mel_2.png')
   mel_3 = plot_mel(mel_3.squeeze().numpy())
   mel_3.savefig('/tmp/mel_3.png')

   print(loss_1)
   print(loss_2)


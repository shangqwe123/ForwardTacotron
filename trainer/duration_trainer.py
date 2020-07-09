import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import BCELoss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.duration_predictor import DurationPredictorModel
from models.forward_tacotron import ForwardTacotron, DurationPredictor
from trainer.common import Averager, TTSSession, MaskedL1, LogL1
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.display import stream, simple_table, plot_mel
from utils.paths import Paths


class DurationTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.duration_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model: DurationPredictorModel, optimizer: Optimizer) -> None:
        for i, session_params in enumerate(hp.forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='forward')
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: DurationPredictorModel,
                      optimizer: Optimizer, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (x, m, ids, lens, dur) in enumerate(session.train_set, 1):

                start = time.time()
                model.train()
                x, m, dur, lens = x.to(device), m.to(device), dur.to(device), lens.to(device)

                dur_hat = model(x, dur)
                dur_hat = dur_hat.transpose(1, 2)
                #print(f'dur hat shape {dur_hat.shape}')
                #print(f'dur shape {dur.shape}')
                dur_loss = F.cross_entropy(dur_hat, dur)

                loss = dur_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                optimizer.step()
                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.duration_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('duration', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            dur_val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Duration_Loss/val', dur_val_loss, model.get_step())
            save_checkpoint('duration', self.paths, model, optimizer, is_silent=True)
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: DurationPredictor, val_set: Dataset) -> float:
        model.eval()
        dur_val_loss = 0
        device = next(model.parameters()).device
        for i, (x, m, ids, lens, dur) in enumerate(val_set, 1):
            x, m, dur, lens = x.to(device), m.to(device), dur.to(device), lens.to(device)
            with torch.no_grad():
                dur_hat = model(x, dur)
                dur_hat = dur_hat.transpose(1, 2)
                dur_loss = F.cross_entropy(dur_hat, dur)
                dur_val_loss += dur_loss.item()
        return dur_val_loss / len(val_set)
import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import show_message, load_latest_checkpoint


class Logger(object):
    def __init__(self, config, rank=0):
        self.rank = rank
        self.summary_writer = None
        self.continue_training = config.training_config.continue_training
        self.logdir = config.training_config.logdir
        self.sample_rate = config.data_config.sample_rate

        if self.rank == 0:
            if not self.continue_training and os.path.exists(self.logdir):
                raise RuntimeError(
                    f"You're trying to run training from scratch, "
                    f"but logdir `{self.logdir} already exists. Remove it or specify new one.`"
                )
            if not self.continue_training:
                os.makedirs(self.logdir)
                validation_loss_txt_dummy = open(self.logdir + "/validation_loss.txt", 'w')
                validation_loss_txt_dummy.close()
            self.summary_writer = SummaryWriter(self.logdir)
            self.save_model_config(config)
            validation_loss_txt_dummy = open(self.logdir + "/validation_loss.txt", 'a')
            validation_loss_txt_dummy.close()

    def _log_losses(self, iteration_or_epoch, loss_stats: dict):
        for key, value in loss_stats.items():
            self.summary_writer.add_scalar(key, value, iteration_or_epoch)

    def log_images(self, epoch, image_buf, image_name):
        self.summary_writer.add_image(image_name, image_buf, epoch)

    def log_training(self, iteration, stats, verbose=False):
        if self.rank != 0: return
        stats = {f'training/{key}': value for key, value in stats.items()}
        self._log_losses(iteration, loss_stats=stats)
        show_message(
            f'Iteration: {iteration} | Losses: [{", ".join(f"{value:.4e}" for value in stats.values())}]',
            verbose=verbose
        )

    def log_validation(self, epoch, stats, verbose=True):
        if self.rank != 0: return
        stats = {f'validation/{key}': value for key, value in stats.items()}
        self._log_losses(epoch, loss_stats=stats)
        show_message(
            f'Epoch: {epoch} | Losses: [{", ".join(f"{value:.4e}" for value in stats.values())}]',
            verbose=verbose
        )
        with open(self.logdir + "/validation_loss.txt", 'a') as validation_loss_txt:
            validation_loss_txt.write(f'Epoch: {epoch} | Losses: {[value for value in stats.values()]}\n')

    def save_model_config(self, config):
        if self.rank != 0: return
        with open(f'{self.logdir}/config.json', 'w') as f:
            json.dump(config.to_dict_type(), f)

    def save_checkpoint(self, epoch, model, optimizer=None):
        if self.rank != 0: return
        d = {}
        d['epoch'] = epoch
        d['model'] = model.state_dict()
        if not isinstance(optimizer, type(None)):
            d['optimizer'] = optimizer.state_dict()
        filename = f'{self.summary_writer.log_dir}/checkpoint_{epoch}.pt'
        torch.save(d, filename)

    def load_latest_checkpoint(self, model, optimizer=None):
        if not self.continue_training:
            raise RuntimeError(
                f"Trying to load the latest checkpoint from logdir {self.logdir}, "
                f"but did not set `continue_training=true` in configuration."
            )
        model, optimizer, epoch = load_latest_checkpoint(self.logdir, model, optimizer)
        return model, optimizer, epoch

import os
import argparse
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm

from logger import Logger
from model import Echo2Reverb
from data import RIRDataset
from utils import ConfigWrapper, show_message, str2bool


def run_training(rank, config, args):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus, config.dist_config)
        torch.cuda.set_device(f'cuda:{rank}')

    show_message('Initializing logger...', verbose=args.verbose, rank=rank)
    logger = Logger(config, rank=rank)

    show_message('Initializing model...', verbose=args.verbose, rank=rank)
    model = Echo2Reverb(config).cuda()
    model.init_flags(export_misc=False)
    show_message(f'Number of Echo2Reverb parameters: {model.nparams}', verbose=args.verbose, rank=rank)

    show_message('Initializing optimizer, scheduler and losses...', verbose=args.verbose, rank=rank)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.training_config.lr,
        betas=(config.training_config.scheduler_beta_1, config.training_config.scheduler_beta_2)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training_config.scheduler_step_size,
        gamma=config.training_config.scheduler_gamma
    )
    if config.training_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    show_message('Initializing data loaders...', verbose=args.verbose, rank=rank)
    train_dataset = RIRDataset(config, stage='train')
    train_sampler = DistributedSampler(train_dataset) if args.n_gpus > 1 else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=1
    )

    if rank == 0:
        valid_dataset = RIRDataset(config, stage='validation')
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.training_config.valid_batch_size,
            num_workers=1
        )

    if config.training_config.continue_training:
        show_message('Loading latest checkpoint to continue training...', verbose=args.verbose, rank=rank)
        model, optimizer, epoch_ = logger.load_latest_checkpoint(model, optimizer)
        epoch_size = len(train_dataset) // config.training_config.batch_size
        epoch_start = epoch_ + 1
        iteration = epoch_size * (epoch_ + 1)
    else:
        iteration = 0
        epoch_start = 0

    if args.n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        show_message(f'INITIALIZATION IS DONE ON RANK {rank}.')

    melspec_loss_list = [
        torchaudio.transforms.MelSpectrogram(
            sample_rate=config.data_config.sample_rate,
            n_fft=fft_size,
            hop_length=hop_size
        ).cuda() for fft_size, hop_size in zip(
            config.model_config.fft_sizes_for_compute_loss,
            config.model_config.hop_sizes_for_compute_loss
        )
    ]

    show_message('Start training...', verbose=args.verbose, rank=rank)
    try:
        for epoch in range(epoch_start, config.training_config.n_epoch):
            # Training step
            model.train()
            for batch in (
                    tqdm(train_dataloader, leave=False)
                    if args.verbose and rank == 0 else train_dataloader
            ):
                model.zero_grad()
                batch = batch.cuda()
                batch = batch + 1e-5 * torch.randn_like(batch)
                input_batch = batch[..., :config.model_config.rir_early_length]
                if config.training_config.use_fp16:
                    with torch.cuda.amp.autocast():
                        loss = (model if args.n_gpus == 1 else model.module).compute_loss(
                            rir_ground_truth=batch,
                            input_data=input_batch,
                            mel_spec_func_list=melspec_loss_list
                        )
                        scaler.scale(loss['total_loss']).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss = (model if args.n_gpus == 1 else model.module).compute_loss(
                        rir_ground_truth=batch,
                        input_data=input_batch,
                        mel_spec_func_list=melspec_loss_list
                    )
                    loss['total_loss'].backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=config.training_config.grad_clip_threshold
                )

                if config.training_config.use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                loss_stats = {f"{key}": value.item() for key, value in loss.items()}
                loss_stats['grad_norm'] = grad_norm.item()
                logger.log_training(iteration, loss_stats, verbose=False)

                iteration += 1
                del loss

            # Validation step after epoch on rank==0 GPU
            if epoch % config.training_config.validation_interval == 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    # Calculating validation set loss
                    del(loss_stats['grad_norm'])
                    valid_loss = {f"{key}": 0 for key, _ in loss_stats.items()}
                    for i, batch in enumerate(
                        tqdm(valid_dataloader) \
                        if args.verbose and rank == 0 else valid_dataloader
                    ):
                        batch = batch.cuda()    # [1, 1, audio_length]
                        batch = batch + 1e-5 * torch.randn_like(batch)
                        input_batch = batch[..., :config.model_config.rir_early_length]
                        valid_loss_ = (model if args.n_gpus == 1 else model.module).compute_loss(
                            rir_ground_truth=batch,
                            input_data=input_batch,
                            mel_spec_func_list=melspec_loss_list
                        )
                        for key, _ in valid_loss.items():
                            valid_loss[key] += valid_loss_[key]
                    for key, _ in valid_loss.items():
                        valid_loss[key] /= (i + 1)
                    loss_stats = {f"{key}": value.item() for key, value in valid_loss.items()}
                    logger.log_validation(epoch, loss_stats, verbose=args.verbose)

                logger.save_checkpoint(
                    epoch,
                    model if args.n_gpus == 1 else model.module,
                    optimizer
                )
            # Learning rate scheduling
            if epoch % (epoch // 10 + 1) == 0:
                scheduler.step()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: training has been stopped.')
        cleanup()
        return


def run_distributed(fn, config, args):
    try:
        mp.spawn(fn, args=(config, args), nprocs=args.n_gpus, join=True)
    except:
        cleanup()


def init_distributed(rank, n_gpus, dist_config):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = dist_config.MASTER_PORT

    torch.distributed.init_process_group(backend='nccl', world_size=n_gpus, rank=rank)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='configuration file')
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if args.n_gpus > 1:
        run_distributed(run_training, config, args)
    else:
        run_training(0, config, args)

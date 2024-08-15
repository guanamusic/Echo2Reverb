import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from model import Echo2Reverb
from utils import ConfigWrapper, show_message, str2bool, parse_filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', required=True,
        type=str, help='configuration file path'
    )
    parser.add_argument(
        '-ch', '--checkpoint_path',
        required=True, type=str, help='checkpoint path'
    )
    parser.add_argument(
        '-rs', '--rir_filelist', required=True, type=str,
        help='RIR filelist, files of which should be just a torch.Tensor array of shape [1, T]'
    )
    parser.add_argument(
        '-exp', '--export_misc', required=False, type=str2bool,
        nargs='?', const=True, default=False, help='export misc stuffs'
    )
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    # Initializing config
    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    # Initializing the model
    model = Echo2Reverb(config)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'], strict=False)
    model.init_flags(export_misc=args.export_misc)

    # Trying to run inference on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Preparing to inference
    melspec_input = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_fft=1024,
        hop_length=128
    ).to(device)
    sample_rate = config.data_config.sample_rate
    rir_length = int(config.data_config.rir_length_in_ms * sample_rate / 1000)

    checkpoint_name = os.path.basename(args.checkpoint_path).replace('.pt', '')
    dir_name = args.checkpoint_path.split('/')[-2]

    save_dir = \
        f'generated/{dir_name}_{checkpoint_name}_withMiscValues' \
        if args.export_misc else f'generated/{dir_name}_{checkpoint_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(f"{save_dir}/misc_values") if args.export_misc else None

    # Inference
    filelist = parse_filelist(args.rir_filelist)
    inference_times = []
    for rir_path in (tqdm(filelist, leave=False) if args.verbose else filelist):
        with torch.no_grad():
            model.eval()
            rir, sr = torchaudio.load(rir_path)
            rir = rir.to(device)

            if rir.size(1) < rir_length:
                rir = torch.nn.functional.pad(rir, (0, rir_length - rir.size(1)), mode='constant')
            rir = rir[..., :rir_length].unsqueeze(0)
            rir = rir + 1e-5 * torch.randn_like(rir)
            input_data = rir[..., :config.model_config.rir_early_length]

            start = datetime.now()
            outputs = model.forward(input_data=input_data)
            end = datetime.now()

            rir_output = outputs[0].cpu().squeeze()

            baseidx = os.path.basename(os.path.abspath(rir_path))
            save_path = f'{save_dir}/generated_{baseidx}'
            torchaudio.save(save_path, rir_output.unsqueeze(0), sample_rate)
            if args.export_misc:
                baseidx_misc = baseidx.replace('.wav', '.pt')
                torch.save(outputs[-1], f'{save_dir}/misc_values/misc_{baseidx_misc}')

            inference_time = (end - start).total_seconds()
            inference_times.append(inference_time)

    show_message(
        f'Done. Inference time estimate: {np.mean(inference_times)} ± {np.std(inference_times)}',
        verbose=args.verbose
    )

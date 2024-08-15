import torch
import torchaudio

from utils import parse_filelist


class RIRDataset(torch.utils.data.Dataset):
    """
    Provides early RIR dataset management for given filelist.
    """
    def __init__(self, config, stage='train'):
        super(RIRDataset, self).__init__()
        self.config = config
        self.stage = stage
        self.sample_rate = config.data_config.sample_rate
        self.rir_total_samples = int(config.data_config.rir_length_in_ms * self.sample_rate / 1000)

        if self.stage == 'train':
            self.filelist_path = config.training_config.train_audio_filelist_path
        elif self.stage == 'validation':
            self.filelist_path = config.training_config.valid_audio_filelist_path
        elif self.stage == 'test':
            self.filelist_path = config.training_config.test_audio_filelist_path
        else:
            raise Exception(f"The dataset stage should be 'train', 'validation', or 'test', not '{self.stage}'.")

        self.rir_paths = parse_filelist(self.filelist_path)

    def load_audio_to_torch(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)
        assert len(audio.size()) == 2
        assert audio.size(0) == 1, f"The channel # of the audio should be 1, not {audio.size(0)}"

        # Zero-padding when audio_samples < self.rir_total_samples
        if audio.size(1) < self.rir_total_samples:
            audio = torch.nn.functional.pad(audio, (0, self.rir_total_samples - audio.size(1)), mode='constant')

        audio = audio[:, :self.rir_total_samples]

        return audio, sample_rate

    def __getitem__(self, index):
        rir_path = self.rir_paths[index]
        rir, sample_rate = self.load_audio_to_torch(rir_path)
        assert sample_rate == self.sample_rate, \
            f"Got path to audio of sampling rate {sample_rate}, but required {self.sample_rate} according config."

        return rir.data

    def __len__(self):
        return len(self.rir_paths)

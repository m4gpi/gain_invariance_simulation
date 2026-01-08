import pytest

from src.core.data.sine_wave import SineWaveDataset, SineWaveDataModule
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram

def test_sine_wave_dataset():
    transforms = LogMelSpectrogram(sample_rate=48_000, num_mel_bins=64, window_length=512, hop_length=384)
    data = SineWaveDataset(N=4096, sample_rate=48_000, transforms=transforms)
    import code; code.interact(local=locals())
    data[0]

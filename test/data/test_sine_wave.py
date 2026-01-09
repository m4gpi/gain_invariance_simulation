import pytest
import numpy as np

from matplotlib import pyplot as plt

from src.core.data.sine_wave import SineWaveDataset, SineWaveDataModule
from src.core.transforms.log_mel_spectrogram import LogMelSpectrogram
from src.core.utils.sketch import plot_mel_spectrogram

def test_sine_wave_dataset():
    data = SineWaveDataset(N=4096, sample_rate=48_000)
    fig = plt.figure()
    x = data[0]
    plt.plot(x)
    fig.savefig("test_wav.png")

    spectrogram = LogMelSpectrogram(sample_rate=48_000, num_mel_bins=64, window_length=512, hop_length=384)
    data = SineWaveDataset(N=4096, sample_rate=48_000, transforms=spectrogram)
    x = data[0]
    fig = plt.figure()
    plot_mel_spectrogram(
        20 * np.log10(x.exp().squeeze().numpy().T),
        sample_rate=spectrogram.sample_rate,
        hop_length=spectrogram.hop_length,
        num_mel_bins=spectrogram.num_mel_bins,
        mel_max_hertz=spectrogram.mel_max_hertz,
        mel_min_hertz=spectrogram.mel_min_hertz,
        mel_break_frequency=spectrogram.mel_break_frequency,
        mel_scaling_factor=spectrogram.mel_scaling_factor,
    )
    fig.savefig("test_spec.png")


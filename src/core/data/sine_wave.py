import attrs
import lightning as L
import torch

from typing import Any, Callable, Dict, List, Tuple

@attrs.define()
class SineWaveDataset(torch.utils.data.Dataset):
    N: int = attrs.field(default=4096)
    sample_rate: int = attrs.field(default=48_000)
    duration_seconds: float = attrs.field(default=1.536)
    min_freq: int = attrs.field(default=100)

    transforms: Callable | None = attrs.field(default=None)
    x: torch.Tensor = attrs.field(init=False)

    def __len__(self):
        return N

    def __getitem__(self, idx):
        if not self.transforms:
            return self.x[idx]
        return self.transforms(self.x[idx])

    def __attrs_post_init__(self):
        A = 1
        f = torch.linspace(self.min_freq, self.sample_rate // 2, self.N)
        t = self.duration_seconds * self.sample_rate
        self.x = A * torch.sin(2 * torch.pi * f * t)

@attrs.define()
class SineWaveDataModule(L.LightningDataModule):
    N: int = attrs.field(default=4096)
    sample_rate: int = attrs.field(default=48_000)
    duration_seconds: float = attrs.field(default=1.536)

    transforms: List[Callable] = attrs.field(default=None)
    train_batch_size: int | None = attrs.field(default=None)
    eval_batch_size: int | None = attrs.field(default=None)
    val_prop: float = attrs.field(default=0.2, validator=attrs.validators.instance_of(float))
    test_prop: float = attrs.field(default=0.2, validator=attrs.validators.instance_of(float))

    seed: int = attrs.field(default=None)
    num_workers: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))
    persist_workers: bool | None = attrs.field(default=None)
    pin_memory: bool = attrs.field(default=True, validator=attrs.validators.instance_of(bool))

    generator: torch.Generator = attrs.field(init=False)

    data: torch.utils.data.Dataset = attrs.field(init=False)
    train_data: torch.utils.data.Dataset = attrs.field(init=False)
    val_data: torch.utils.data.Dataset = attrs.field(init=False)
    test_data: torch.utils.data.Dataset = attrs.field(init=False)

    def setup(self):
        data = SineWaveDataset(N=self.N, sample_rate=self.sample_rate, duration_seconds=self.duration_seconds)
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(data, (1 - self.val_prop - self.test_prop, self.val_prop, self.test_prop))
        return self

    def train_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.train_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.val_data, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def predict_dataloader(self, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        batch_size = batch_size or self.eval_batch_size or len(self.test_data)
        return self._build_dataloader(self.test_data, batch_size=self.eval_batch_size, shuffle=False)

    def _build_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None, **kwargs: Any) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or len(dataset),
            collate_fn=self.batch_converter,
            **self.dataloader_params,
            **kwargs
        )

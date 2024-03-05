import pickle
import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import json
import torchvision
from torch import Tensor
from typing import Tuple
import matplotlib.pyplot as plt

from easydict import EasyDict

class Kinetics_Dataset(torchvision.datasets.Kinetics):

    def __init__(self, config, mode='train', metadata=None):
        self.config = config
        print(mode)
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                # transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        super().__init__(self.config.dataset.data_root,
                         frames_per_clip= 16,
                         step_between_clips = 100,
                         frame_rate = 2,
                         _precomputed_metadata=metadata,
                         _audio_channels=0,
                         transform=self.transform,
                         num_workers=3,
                         num_classes="400",
                         split=mode,
                         download=False)

    def process_audio(self, audio, prev_audio_fps, resamp_audio_fps):
        secs = len(audio) / prev_audio_fps  # Number of seconds in signal X
        samps = secs * resamp_audio_fps  # Number of samples to downsample
        resamples = signal.resample(audio, int(samps))
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        frequencies, times, spectrogram = signal.spectrogram(resamples, resamp_audio_fps, nperseg=512, noverlap=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        if self.config.dataset.pad_audio:
            diff = self.config.dataset.pad_audio - spectrogram.shape[1]
            if diff>0:
                spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, diff)), mode='constant', constant_values=spectrogram.min())

        return spectrogram

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            new_video = torch.cat([self.transform(video[i]).unsqueeze(dim=0) for i in range(len(video))])
            video = new_video
        video = video.permute(1, 0 , 2, 3)

        audio = self.process_audio(audio[0], info["audio_fps"], 16000)


        return {"data":{1:video, 2:audio},"label": label}

class Kinetics_Dataloader():

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val, dataset_test = self._get_datasets()

        self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                                        batch_size=self.config.training_params.batch_size,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory,
                                                        worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id))
        self.valid_loader = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=self.config.training_params.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=self.config.training_params.data_loader_workers,
                                                        pin_memory=self.config.training_params.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=self.config.training_params.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=self.config.training_params.data_loader_workers,
                                                       pin_memory=self.config.training_params.pin_memory)



    def _get_datasets(self):
        file = open('./datasets/Kinetics/metadata_train_16_2.pkl', 'rb')
        metadata_train = pickle.load(file)
        file = open('./datasets/Kinetics/metadata_val_16_2.pkl', 'rb')
        metadata_val = pickle.load(file)
        file = open('./datasets/Kinetics/metadata_test_16_2.pkl', 'rb')
        metadata_test = pickle.load(file)
        del file

        train_dataset = Kinetics_Dataset(config=self.config, mode="train", metadata=metadata_train)
        test_dataset = Kinetics_Dataset(config=self.config, mode="val", metadata=metadata_val)
        valid_dataset = Kinetics_Dataset(config=self.config, mode="test", metadata=metadata_test)

        return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    config_filename = "/users/sista/kkontras/Documents/Balance/configs/Kinetics/audio.json"
    with open(config_filename, 'r') as config_json:
        a = json.load(config_json)
        config = EasyDict(a)

    val_dataset = Kinetics_Dataset(config = config, mode="val", metadata=None)

    for batch in val_dataset:

        fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(24, 6))

        axes = axes.flatten()

        for i, img in enumerate(batch["data"][1]):
            img = img.numpy().transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis('off')  # Turn off axis

        plt.tight_layout()
        plt.show()
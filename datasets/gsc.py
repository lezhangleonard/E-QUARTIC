import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import requests
from io import BytesIO
from tarfile import TarFile
from scipy.io import wavfile
from scipy.signal import resample
import librosa
from scipy.ndimage import zoom
from datasets.datasets import *
import torchvision.transforms as transforms


class AddRandomNoise(torch.nn.Module):
    def __init__(self, noise_level=0.005):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x):
        noise = torch.randn(x.size()) * self.noise_level
        return x + noise

class TimeMasking(torch.nn.Module):
    def __init__(self, max_mask_size=10):
        super().__init__()
        self.max_mask_size = max_mask_size

    def forward(self, x):
        mask_size = np.random.randint(0, self.max_mask_size)
        start = np.random.randint(0, x.size(2) - mask_size)
        x[:, :, start:start+mask_size] = 0
        return x

class FrequencyMasking(torch.nn.Module):
    def __init__(self, max_mask_size=5):
        super().__init__()
        self.max_mask_size = max_mask_size

    def forward(self, x):
        mask_size = np.random.randint(0, self.max_mask_size)
        start = np.random.randint(0, x.size(1) - mask_size)
        x[:, start:start+mask_size, :] = 0
        return x
    
spectrogram_transforms = transforms.Compose([
    transforms.ToTensor(),
    AddRandomNoise(),
    TimeMasking(),
    FrequencyMasking(),
])

spectrogram_transforms_test = transforms.Compose([
    transforms.ToTensor(),
])


class SpeechCommandsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def download_and_extract_speech_commands(url: str, extract_to: str = './datasets/SpeechCommands'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print('Downloading and extracting dataset')
        response = requests.get(url, stream=True)
        tar = TarFile.open(fileobj=BytesIO(response.raw.read()), mode='r:gz')
        tar.extractall(path=extract_to)
    else:
        print('Dataset already downloaded and extracted')

def load_speech_commands_data(directory: str, resample_to: int = 8000, n_fft: int = 256, hop_length: int = 64):
    labels = []
    data = []
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    target_shape = (41, 64)

    if os.path.exists(os.path.join(directory, 'data.pt')) and os.path.exists(os.path.join(directory, 'labels.pt')):
        data_tensor = torch.load(os.path.join(directory, 'data.pt'))
        labels_tensor = torch.load(os.path.join(directory, 'labels.pt'))
        return data_tensor, labels_tensor

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Please check the dataset path.")
        return None, None

    words_set = set(words)
    word_to_index = {word: i for i, word in enumerate(words)}
    num_labels = len(words)

    for word in words_set:
        word_dir = os.path.join(directory, word)
        if os.path.isdir(word_dir):
            for filename in os.listdir(word_dir):
                if filename.endswith('.wav'):
                    filepath = os.path.join(word_dir, filename)
                    sample_rate, audio = wavfile.read(filepath)
                    if sample_rate != resample_to:
                        audio = resample(audio, int(len(audio) * resample_to / sample_rate))
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    stft_audio = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)).T
                    zoom_factors = (target_shape[0] / stft_audio.shape[0], target_shape[1] / stft_audio.shape[1])
                    stft_audio_resized = zoom(stft_audio, zoom_factors)
                    stft_audio = stft_audio_resized / np.max(stft_audio_resized)
                    data.append(stft_audio)
                    label = np.zeros(num_labels)
                    label[word_to_index[word]] = 1
                    labels.append(label)
    if not data:
        print("No audio data found. Please check if the dataset is loaded correctly.")
        return None, None
    max_length = max(len(a) for a in data)
    data = [np.pad(a, ((0, max_length - len(a)), (0, 0)), 'constant') for a in data]
    data_tensor = torch.tensor(np.stack(data), dtype=torch.float32)
    data_tensor = data_tensor.unsqueeze(1)
    labels_tensor = torch.tensor(np.stack(labels), dtype=torch.float32).argmax(dim=1)
    torch.save(data_tensor, os.path.join(directory, 'data.pt'))
    torch.save(labels_tensor, os.path.join(directory, 'labels.pt'))
    print(data_tensor.shape, labels_tensor.shape)
    return data_tensor, labels_tensor

def load_test_data(batch_size: int) -> DataLoader:
    download_and_extract_speech_commands('http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz', './datasets/SpeechCommands/test')
    data, labels = load_speech_commands_data('./datasets/SpeechCommands/test')
    test_data, test_labels = data, labels
    test_dataset = SpeechCommandsDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def load_train_val_data(batch_size: int, train_val_split: float, weighted: bool=False, shuffle: bool=True) -> (DataLoader, DataLoader):
    download_and_extract_speech_commands('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', './datasets/SpeechCommands/train')
    data, labels = load_speech_commands_data('./datasets/SpeechCommands/train')
    dataset = SpeechCommandsDataset(data, labels)
    trainset, valset = torch.utils.data.random_split(dataset, [int(train_val_split*len(dataset)), len(dataset)-int(train_val_split*len(dataset))])
    if weighted:
        train_data, train_targets = extract_data_targets(trainset, dataset)
        trainset = WeightedDataset(train_data, train_targets, alpha=0.01, transform=spectrogram_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader, Dataset
from src.tokenizer import ASREncoderDecoder
import warnings
warnings.filterwarnings('ignore')


data_dir = './data'
tokenizer = ASREncoderDecoder()


class CustomLibriSpeech(Dataset):
    def __init__(self, samples_dir, labels_dir):
        self.sample_paths = sorted(
            [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith('.pt')]
        )
        self.label_paths = sorted(
            [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.pt')]
        )
    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        # Load spectrogram
        spectrogram = torch.load(self.sample_paths[idx])
        # Load label
        label = torch.load(self.label_paths[idx])
        # Calculate input and label lengths
        input_length = spectrogram.shape[0] // 2
        label_length = len(label)
        return spectrogram, label, input_length, label_length


def preprocess_and_save_data(subset, samples_dir, labels_dir):
    print(f'Downloading {subset}...')
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=data_dir,
        url=subset,
        download=True
    )

    mfcc_transform = MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 40,
            'center': False
        }
    )

    print(f'Processing and saving {subset}...')
    for idx, (waveform, _, label, _, _, _) in enumerate(dataset):
        # Process spectrogram
        spec = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
        sample_path = os.path.join(samples_dir, f'sample_{idx}.pt')
        torch.save(spec, sample_path)
        label_tensor = torch.Tensor(tokenizer.encode(label.lower()))
        label_path = os.path.join(labels_dir, f'label_{idx}.pt')
        torch.save(label_tensor, label_path)


def get_dataset(subset, samples_dir, labels_dir):
    if not os.listdir(samples_dir) or not os.listdir(labels_dir):
        preprocess_and_save_data(subset, samples_dir, labels_dir)
    return CustomLibriSpeech(samples_dir, labels_dir)


def get_dataloader(subset, batch_size, samples_dir, labels_dir): 
    def collate_fn(batch):
        spectrograms, labels, input_lengths, label_lengths = zip(*batch)
        padded_spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)
        
        return padded_spectrograms, padded_labels, input_lengths, label_lengths

    return DataLoader(
        dataset=get_dataset(subset, samples_dir, labels_dir),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def get_train_dataloader(batch_size):
    # Create directories
    train_samples_dir = os.path.join(data_dir, 'train_samples')
    train_labels_dir = os.path.join(data_dir, 'train_labels')
    os.makedirs(train_samples_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    subset = "train-clean-100"
    return get_dataloader(subset, batch_size, train_samples_dir, train_labels_dir)


def get_validation_dataloader(batch_size):
    val_samples_dir = os.path.join(data_dir, 'val_samples')
    val_labels_dir = os.path.join(data_dir, 'val_labels')
    os.makedirs(val_samples_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    subset = "dev-clean"
    return get_dataloader(subset, batch_size, val_samples_dir, val_labels_dir)


if __name__ == '__main__':
    # Example usage
    train_dataloader = get_train_dataloader(16)
    val_dataloader = get_validation_dataloader(16)

    # Check training data
    print("Training Data:")
    for spectrograms, labels, input_lengths, label_lengths in train_dataloader:
        print(spectrograms.shape)
        break

    # Check validation data
    print("Validation Data:")
    for spectrograms, labels, input_lengths, label_lengths in val_dataloader:
        print(spectrograms.shape)
        break

import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader, Dataset
from src.tokenizer import ASREncoderDecoder
import warnings
warnings.filterwarnings('ignore')


data_dir = './data'
samples_dir = os.path.join(data_dir, 'samples')
labels_dir = os.path.join(data_dir, 'labels')
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

tokenizer = ASREncoderDecoder()

class CustomLibriSpeech(Dataset):
    def __init__(self):
        # Automatically load file paths for samples and labels
        self.sample_paths = sorted(
            [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith('.pt')]
        )
        self.label_paths = sorted(
            [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.pt')]
        )

        if len(self.sample_paths) != len(self.label_paths):
            raise ValueError("Mismatch between the number of samples and labels.")

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


def preprocess_and_save_data(subset):
    print('Downloading Data...')
    train_dataset = torchaudio.datasets.LIBRISPEECH(
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

    print('Processing and saving data...')
    for idx, (waveform, _, label, _, _, _) in enumerate(train_dataset):
        # Process spectrogram
        spec = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
        sample_path = os.path.join(samples_dir, f'sample_{idx}.pt')
        torch.save(spec, sample_path)

        # Process label
        label_tensor = torch.Tensor(tokenizer.encode(label.lower()))
        label_path = os.path.join(labels_dir, f'label_{idx}.pt')
        torch.save(label_tensor, label_path)


def get_dataset(subset):
    # Preprocess data and save if directories are empty
    if not os.listdir(samples_dir) or not os.listdir(labels_dir):
        preprocess_and_save_data(subset)
    return CustomLibriSpeech()


def get_dataloader(subset, batch_size): 
    def collate_fn(batch):
        spectrograms, labels, input_lengths, label_lengths = zip(*batch)
        padded_spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)
        
        return padded_spectrograms, padded_labels, input_lengths, label_lengths

    return DataLoader(
        dataset=get_dataset(subset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

if __name__ == '__main__':
    # Example of usage
    subset = "train-clean-100"
    dataloader = get_dataloader(subset, 16)
    for (spectrograms, labels, input_lengths, label_lengths) in dataloader:
        print(spectrograms.shape)
        break
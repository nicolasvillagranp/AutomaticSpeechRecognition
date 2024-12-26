import torch
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader, Dataset
from src.tokenizer import ASREncoderDecoder



data_dir = './data'
tokenizer = ASREncoderDecoder()

class CustomLibriSpeech(Dataset):
    def __init__(self, spectograms, labels, input_lengths, label_lengths):
        self.spectograms = spectograms
        self.labels = labels
        self.input_lengths = input_lengths
        self.label_lengths = label_lengths
    def __len__(self):
        return len(self.spectograms)
    def __getitem__(self, idx):
        return (self.spectograms[idx], 
                self.labels[idx], 
                self.input_lengths[idx],
                self.label_lengths[idx]
                )

def get_dataset(subset):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    print('Downloading Data...')
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        root=data_dir,
        url=subset,
        download=True
    )

    mfcc_transform = MFCC(
        sample_rate=16000, # Sample rate of mfcc transform
        n_mfcc=40,
        melkwargs={
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 40,
            'center': False
        }
    )

    print('Processing Data....')
    for (waveform, _, labels, _, _, _) in train_dataset:
        spec = mfcc_transform(waveform).squeeze(0).transpose(0,1)
        spectrograms.append(spec)
        label = torch.Tensor(tokenizer.encode(labels.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    return CustomLibriSpeech(spectrograms, labels, input_lengths, label_lengths)


def get_dataloader(subset, batch_size): 
    def collate_fn(batch):
        spectograms, labels, input_lengths, label_lengths = zip(*batch)
        padded_spectograms = torch.nn.utils.rnn.pad_sequence(spectograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)
        
        return padded_spectograms, padded_labels, input_lengths, label_lengths

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
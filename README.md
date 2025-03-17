# Automatic Speech Recognition (ASR) Model

## Overview
This project implements an Automatic Speech Recognition (ASR) system using PyTorch and torchaudio. It leverages the LibriSpeech dataset, applies MFCC transformations, and trains a neural network for speech-to-text transcription using the CTC (Connectionist Temporal Classification) loss.

## Features
- Downloads and preprocesses LibriSpeech dataset.
- Uses MFCC feature extraction.
- Implements a ResNet-based neural network with bidirectional GRU layers.
- Provides functions for training, validation, and inference.
- Uses a tokenizer for character-based encoding and decoding.
- Supports beam search decoding for improved transcription accuracy.

## Run 
- Run data.py to download data
- Run train.py to train a model
- Run validate.py to validate a model

from torchaudio.functional import edit_distance
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import ctc_decoder_config
from src.data import get_validation_dataloader
from src.model import ASRModel, load_model
from src.tokenizer import ASREncoderDecoder
import torch


def validate_model(model, dataloader, tokenizer):
    # Initialize the decoder for beam search
    decoder_config = ctc_decoder_config(
        beam_width=5,  # Number of beams
        blank_id=0,  # CTC blank id
        logits_length_beam_prune=30.0, 
    )
    decoder = ctc_decoder(decoder_config)

    model.eval()
    total_wer = 0.0
    total_sentences = 0

    with torch.no_grad():
        for spectrograms, labels, input_lengths, label_lengths in dataloader:
            spectrograms = spectrograms.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = labels.to(spectrograms.device)
            input_lengths = input_lengths.to(spectrograms.device)
            label_lengths = label_lengths.to(spectrograms.device)

            logits = model(spectrograms) 
            logits = logits.transpose(0, 1)  

            # Beam Search
            decoded_outputs = decoder(logits)
            decoded_sentences = [tokenizer.decode(output[0]) for output in decoded_outputs]

            # Labels to text
            ground_truth_sentences = [
                tokenizer.decode(labels[i][:label_lengths[i]].tolist())
                for i in range(len(labels))
            ]

            # Calculate WER
            for gt_sentence, pred_sentence in zip(ground_truth_sentences, decoded_sentences):
                total_wer += edit_distance(gt_sentence.split(), pred_sentence.split()) / len(gt_sentence.split())
                total_sentences += 1

    avg_wer = total_wer / total_sentences
    print(f"Average WER: {avg_wer:.4f}")
    return avg_wer


if __name__ == "__main__":
    # Load model
    model = ASRModel(5, 5, 512, 40, 28)
    load_model(model)
    tokenizer = ASREncoderDecoder()
    # Get validation dataloader
    val_dataloader = get_validation_dataloader(batch_size=16)
    # Validate the model
    validate_model(model, val_dataloader, tokenizer)

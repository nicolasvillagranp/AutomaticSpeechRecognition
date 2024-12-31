import torch
from tqdm import tqdm
from torchaudio.functional import edit_distance
from torchaudio.models.decoder import ctc_decoder

from src.data import get_validation_dataloader
from src.model import ASRModel, load_model
from src.tokenizer import ASREncoderDecoder  # your class







def validate_model(model, dataloader, tokenizer):
    """
    Validate the model on a given dataloader using a CTC beam search decoder,
    with a tqdm progress bar to show progress.
    """

    
    tokens = [tokenizer.index_to_char[i] for i in range(len(tokenizer))]
    decoder = ctc_decoder(
        tokens=tokens,
        lexicon=[],
        beam_size=5,           
        beam_threshold=30.0,    
        lm_weight=0.0,        
        word_score=0.0,
        blank_token="<blank>",  # match ASR encoder-decoder
        sil_token=" ",         
        log_add=False,
    )

    model.eval()
    total_wer = 0.0
    total_sentences = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for spectrograms, labels, input_lengths, label_lengths in tqdm(dataloader, desc="Validating", total=len(dataloader)):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)
            logits = model(spectrograms)   
            logits = logits.transpose(0, 1) 
            log_probs = logits.log_softmax(dim=-1)  

            emissions = log_probs.transpose(0, 1) 
            decoded_outputs = decoder(
                emissions.to('cpu').contiguous(),
                input_lengths.to('cpu').contiguous() )
            decoded_sentences = [
                tokenizer.decode(hypotheses[0].tokens) 
                for hypotheses in decoded_outputs
            ]

            ground_truth_sentences = [
                tokenizer.decode(labels[i][:label_lengths[i]].tolist())
                for i in range(len(labels))
            ]
            # Get WER error
            for gt_sentence, pred_sentence in zip(ground_truth_sentences, decoded_sentences):
                total_wer += (
                    edit_distance(gt_sentence.split(), pred_sentence.split())
                    / len(gt_sentence.split())
                )
                total_sentences += 1

    avg_wer = total_wer / total_sentences if total_sentences > 0 else 0.0
    return avg_wer


if __name__ == "__main__":
    model = ASRModel(1, 2, 512, 40, 28)  #
    load_model(model)

    tokenizer = ASREncoderDecoder()
    val_dataloader = get_validation_dataloader(batch_size=16)
    wer = validate_model(model, val_dataloader, tokenizer)
    print(f"Validation WER: {wer:.4f}")


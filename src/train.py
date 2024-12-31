import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.model import ASRModel, save_model, load_model
from src.data import get_train_dataloader, get_validation_dataloader
from src.tokenizer import ASREncoderDecoder
from src.validate import validate_model
torch.manual_seed(7)

def train(model):
    epochs = 100
    tokenizer = ASREncoderDecoder()
    optimizer = torch.optim.AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CTCLoss(blank=0).to(device)
    train_dataloader = get_train_dataloader(32)
    val_dataloader = get_validation_dataloader(32)
    model.to(device)

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            with tqdm(train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                for spectrograms, labels, input_lengths, label_lengths in tepoch:
                    spectrograms, labels = spectrograms.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(spectrograms)  
                    output = F.log_softmax(output, dim=2)
                    output = output.transpose(0, 1)  

                    loss = criterion(output, labels, input_lengths, label_lengths)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
            if epoch % 10 == 0:
                print(f'WER in validation {validate_model(model, val_dataloader, tokenizer)}')
                # Save the model after every epoch
                save_model(model, f"models/asr_model_epoch{epoch + 1}.pth")

            print(f"Epoch {epoch + 1} complete. Average Loss: {epoch_loss / len(train_dataloader):.4f}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        save_model(model)
    finally:
        print("Training complete. Saving final model...")
        save_model(model)

if __name__ == '__main__':
    model = ASRModel(1, 2, 512, 40, 28)
    load_model(model)
    train(model)

    


    
        
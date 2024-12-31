import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.model import ASRModel, save_model, load_model
from src.data import get_dataloader

torch.manual_seed(7)

def train(model):
    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CTCLoss(blank=0).to(device)
    train_dataloader = get_dataloader("train-clean-100", 16)
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

            print(f"Epoch {epoch + 1} complete. Average Loss: {epoch_loss / len(train_dataloader):.4f}")
            # Save the model after every epoch
            save_model(model, f"models/asr_model_epoch{epoch + 1}.pth")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        save_model(model)
    finally:
        print("Training complete. Saving final model...")
        save_model(model)

if __name__ == '__main__':
    model = ASRModel(1, 2, 256, 40, 28)
    # load_model(model)
    train(model)

    


    
        
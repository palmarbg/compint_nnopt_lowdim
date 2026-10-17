import torch.nn as nn
import torch

def _train_one_epoch(model, device, dataloader, criterion, optimizer):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * data.size(0)

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def _validate(model, device, dataloader, criterion):
    """Evaluate the model on a validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=10):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = None

    for epoch in range(epochs):
        train_loss, train_acc = _train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = _validate(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Check for improvement.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping trigger.
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights.
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, train_losses, train_accs, val_losses, val_accs
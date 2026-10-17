import torch
import numpy as np
from .show_imgs import show_imgs

def show_inference(model, dataloader, device, n=5):
    pos_data = np.zeros((n, 28, 28))
    pos_label = np.zeros((n), dtype=np.int64)
    neg_data = np.zeros((n, 28, 28))
    neg_label = np.zeros((n), dtype=np.int64)
    pos_idx = 0
    neg_idx = 0

    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            outputs = model(data.to(device))
            _, predicted = torch.max(outputs, 1)

            mask = (target == predicted.to('cpu'))
            
            for i, result in enumerate(mask):
                if result and pos_idx < n:
                    pos_data[pos_idx] = data[i].reshape(28, 28).numpy()
                    pos_label[pos_idx] = target[i].numpy()
                    pos_idx += 1
                elif not result and neg_idx < n:
                    neg_data[neg_idx] = data[i].reshape(28, 28).numpy()
                    neg_label[neg_idx] = predicted[i].cpu().numpy()
                    neg_idx += 1

            if neg_idx >= n and pos_idx >= n:
                break
            
    show_imgs(pos_data, pos_label)
    show_imgs(neg_data, neg_label)
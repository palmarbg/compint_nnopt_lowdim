import torch
from .show_imgs import show_imgs

def show_dataloader_imgs(dataloader, n=8):
    for X, Y in dataloader:
        show_imgs(X[0:n].reshape(-1,28,28).numpy(), Y[0:n].numpy())
        break
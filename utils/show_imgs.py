import matplotlib.pyplot as plt

def show_imgs(imgs, labels, figsize=(16,16)):
    n = len(labels)
    plt.figure(figsize=figsize)
    for k in range(n):
        img, label = imgs[k], labels[k]
        plt.subplot(1, n, k+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(label)
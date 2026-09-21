import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import autoencoderMLP4Layer

DATA_DIR = './data'   # change if using Colab

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def main():
    print('running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [8]')

    args = argParser.parse_args()

    save_file = args.s
    bottleneck_size = args.z

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = FashionMNIST(DATA_DIR, train=False, download=False, transform=test_transform)

    # Configurations
    N_input = 28 * 28
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.to(device)
    # Evaluation mode
    model.eval()
    # Loss function definition
    criterion = nn.MSELoss()

    # Store one image per class
    images_by_class = {}
    # Iterate through the test set
    for image, label in test_set:
        if label not in images_by_class:
            img = image.view(1, N_input).to(device=device)
            images_by_class[label] = img

        if len(images_by_class) == 10:
            break

    # Loss tracker for each class
    loss_by_class = {}
    with torch.no_grad():
        for label in images_by_class.keys():
            img = images_by_class[label]
            output = model(img)
            loss = criterion(output, img)
            loss_by_class[label] = loss            
            print(f"Class: {CLASSES[label]} | Test Loss: {loss.item()}")

    # Find top 3 worst classes
    top_3_worst = sorted(loss_by_class, key=loss_by_class.get, reverse=True)[:3]
    worst_names = [CLASSES[label] for label in top_3_worst]
    print(f"The top-3 worst reconstructions are labels {top_3_worst} ({', '.join(worst_names)}).")

    # Visualize examples of the top 3 worst classes
    f = plt.figure(figsize=(9, 8))
    
    for i, label in enumerate(top_3_worst):
        with torch.no_grad():
            output = model(images_by_class[label])
            
        # Resize for Matplotlib
        img = images_by_class[label].view(28, 28).cpu()
        output = output.view(28, 28).cpu()
        
        # Plot Input
        f.add_subplot(3, 2, 2 * i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'input: {CLASSES[label]}')
        plt.axis('off')
        
        # Plot Reconstruction
        f.add_subplot(3, 2, 2 * i + 2)
        plt.imshow(output, cmap='gray')
        plt.title('reconstruction')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

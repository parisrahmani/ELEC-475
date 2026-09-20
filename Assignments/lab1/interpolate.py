import argparse

import matplotlib.pyplot as plt
import torch
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

    N_input = 28 * 28
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.to(device)
    model.eval()

    print('enter an index from 0 to', len(test_set) - 1, '(or -1 to quit)')

    while True:
        try:
            idx1 = int(input('first index > '))
            idx2 = int(input('second index > '))
        except ValueError:
            print('not an integer')
            continue

        if idx1 < 0 or idx2 < 0:
            break
        if idx1 >= len(test_set) or idx2 >= len(test_set):
            print('index out of range')
            continue

        label1 = test_set.targets[idx1].item()
        print('label =', label1, '(' + CLASSES[label1] + ')')
        label2 = test_set.targets[idx2].item()
        print('label =', label2, '(' + CLASSES[label2] + ')')

        

        img1 = test_set.data[idx1].type(torch.float32) / 255.0
        img1 = img1.view(1, N_input).to(device=device) 
        img2 = test_set.data[idx2].type(torch.float32) / 255.0
        img2 = img2.view(1, N_input).to(device=device)

        with torch.no_grad():
            z1 = model.encode(img1)
            z2 = model.encode(img2)

        n = 8
        alphas = torch.linspace(0, 1, n)
        z_interp = [(1 - a) * z1 + a * z2 for a in alphas]

        with torch.no_grad():
            all_z = [z1] + z_interp + [z2]
            decoded = [model.decode(z).view(28, 28).cpu() for z in all_z]


        label1 = CLASSES[test_set.targets[idx1].item()]
        label2 = CLASSES[test_set.targets[idx2].item()]

        f = plt.figure(figsize=(20, 3))
        for i, img in enumerate(decoded):
            f.add_subplot(1, len(decoded), i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(label1)
            elif i == len(decoded) - 1:
                plt.title(label2)
            else:
                a = alphas[i - 1].item()
                plt.title(f'a={a:.2f}')
        plt.savefig('interpolation_output.png')
        plt.show()
        


if __name__ == '__main__':
    main()

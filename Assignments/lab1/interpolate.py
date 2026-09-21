import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import autoencoderMLP4Layer

DATA_DIR = './data'   # change if using Colab

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

n = 8


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
            idx1 = int(input('First index > '))
            idx2 = int(input('Second index > '))
        except ValueError:
            print('not an integer')
            continue

        if (idx1 < 0) or (idx2 < 0):
            break
        if (idx1 >= len(test_set)) or (idx2 >= len(test_set)):
            print('index out of range')
            continue

        # Print two labels
        label1 = test_set.targets[idx1].item()
        label2 = test_set.targets[idx2].item()

        # Images should be from different classes
        if label1 == label2:
            continue
        

        print('label 1 =', label1, '(' + CLASSES[label1] + ')')
        print('label_2 =', label2, '(' + CLASSES[label2] + ')')

        # 1. Normalize, flatten image, then move to memory
        img1 = test_set.data[idx1].type(torch.float32) / 255.0
        img2 = test_set.data[idx2].type(torch.float32) / 255.0
        img1 = img1.view(1, N_input).to(device=device)
        img2 = img2.view(1, N_input).to(device=device)

        with torch.no_grad():
            # 2. Pass through the encoder and return bottleneck tensors
            z1 = model.encode(img1)
            z2 = model.encode(img2)

            # 3. Interpolate n new bottleneck tensors
            a = np.linspace(0, 1, n)
            mean_tensors = []
            for i in range(n):
                z = (1 - a[i]) * z1 + a[i] * z2
                mean_tensors.append(z)

            # 4. Pass z1, z2 and interpolated tensors through decoder
            
            final_outputs = []
            # First class
            z1_output = model.decode(z1)
            final_outputs.append(z1_output)
            # Interpolated tensors
            for i in range(n):
                output = model.decode(mean_tensors[i])
                final_outputs.append(output)
            # Second class
            z2_output = model.decode(z2)
            final_outputs.append(z2_output)

            # 5. Reshape to 28 * 28 and move to cpu
            for i in range(n+2):
                final_outputs[i] = final_outputs[i].view(28, 28).cpu()

            # 6. Plot figure
            f = plt.figure(figsize=(20, 3))
            for i in range(n + 2):
                f.add_subplot(1, n+2, i+1)
                plt.imshow(final_outputs[i], cmap="gray")
                plt.axis('off')
                if i == 0:
                    plt.title(CLASSES[label1]) 
                elif i == n + 1:
                    plt.title(CLASSES[label2])
                else:
                    plt.title(f"a = {a[i-1]:.2f}")

            plt.show()

if __name__ == '__main__':
    main()

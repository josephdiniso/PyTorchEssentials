#!/usr/bin/env python3

import time
import argparse
import logging
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from mnist_net import Net

logger = logging.getLogger('training')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def show_image(img):
    """
    Utility function to plot an image
    """
    cv2.imshow("Img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_example(example_data: torch.tensor, example_targets: torch.tensor):
    """
    Utility function to display first six examples in the var 'example_data'
        Args:
            example_data - PyTorch tensor of example images
    """
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig

class TrainNet:
    def __init__(self, model_path, optim_path, epochs, **kwargs):
        self.n_epochs = epochs
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.network = Net()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.momentum)
        # PyTorch accumulates gradients by default
        self.optimizer.zero_grad()

        if model_path and optim_path:
            logger.info("Using pretrained model")
            model_dict = torch.load(model_path)
            self.network.load_state_dict(model_dict)
            optim_dict = torch.load(optim_path)
            self.optimizer.load_state_dict(optim_dict)


        # Uses GPU hardware acceleration if applicable
        if torch.cuda.is_available():
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        self.device = torch.device(dev)
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

        self.train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/Datasets/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=self.batch_size_train, shuffle=True, num_workers=12)

        self.test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/Datasets/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=self.batch_size_test, shuffle=True, num_workers=2)


    def train(self, epoch):
        train_losses = []
        train_counter = []
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(self.train_loader.dataset),
                            100. * batch_idx / len(self.train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
            torch.save(self.network.state_dict(), '/home/jdiniso/Models/mnist.pth')
            torch.save(self.optimizer.state_dict(), '/home/jdiniso/Models/optimizer.pth')


    def test(self):
        test_losses = []
        test_counter = [i*len(self.train_loader.dataset) for i in range(self.n_epochs+1)]
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    
    def run_training(self):
        time_now = time.time()
        self.test()
        for epoch in range(1, self.n_epochs + 1):
            self.train(epoch)
            self.test()
        print("Time took {} seconds".format(time.time()-time_now))


    def test_img(self):
        test = iter(self.test_loader)
        with torch.no_grad():
            while 1:
                test_data = next(test)[0]
                for test_val in test_data:
                    print(test_val.shape)
                    output = self.network(test_val)
                    rand_val = random.randint(0,1000)
                    output_val = output[rand_val].max(dim=0).indices.item()
                    img = test_data[rand_val].numpy().squeeze()
                    blank = np.zeros((40, img.shape[1])).astype(np.float32)
                    img = cv2.vconcat([img, blank])
                    img = cv2.resize(img, (200,200))
                    cv2.putText(img, str(output_val), (80,160), 0, 2, 255)
                    cv2.imshow("img", img)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        cv2.destroyAllWindows()
                    elif k == 27:
                        break

def main():
    parser = argparse.ArgumentParser(description="Neural net for MNIST dataset training")
    parser.add_argument("--model_path", type=str, help="Path to a previous model")
    parser.add_argument("--optim_path", type=str, help="Path to previous optimizer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to run")
    parser.add_argument("--save_data", type=str, required=True, help="Directory to save training and testing data")
    parser.add_argument("--test_path", type=str, help="Path of images to test")
    args = parser.parse_args()

    trainer = TrainNet(**vars(args))
    # trainer.run_training()
    trainer.test_img()
    


if __name__ == "__main__":
    main()


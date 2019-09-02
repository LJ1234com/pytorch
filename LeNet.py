import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils import data
import torch.optim as optim

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    lr = 0.001

    train = datasets.MNIST(
        root = '../../study/data',
        train = True,
        transform=transforms.ToTensor(),
        download=False
    )
    test = datasets.MNIST(
        root= '../../study/data',
        train = False,
        transform=transforms.ToTensor(),
        download=False
    )

    train_loader = data.DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = data.DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False
    )

    lenet5 = LeNet5().to(device)
    print(lenet5)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(lenet5.parameters(), lr=lr)
    #
    # lenet5.train()
    # for epoch in range(5):
    #     for x_tr, y_tr in train_loader:
    #         x_tr = x_tr.to(device)
    #         y_tr = y_tr.to(device)
    #         out = lenet5(x_tr)
    #         loss = criterion(out, y_tr)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    # with torch.no_grad():
    #     correct = 0
    #     for x_te, y_te in test_loader:
    #         x_te = x_te.to(device)
    #         y_te = y_te.to(device)
    #         predict = lenet5(x_te)
    #         correct += (torch.argmax(predict, dim=1) == y_te).sum()
    #     print('{}, {}'.format(correct, len(test)))


'''
https://blog.csdn.net/jeryjeryjery/article/details/79426907
Lenet、Alexnet 、VGG、 GoogleNet、ResNet

'''
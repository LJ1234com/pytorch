import torch
from  torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils import data
from torch import optim
import math


def make_layers(cfg, batch_norm=True):
   in_channel = 3
   layers = []
   for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels=in_channel, out_channels=v, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(num_features=v), nn.ReLU(inplace=True)]
            else:
                layers += [nn.ReLU(inplace=True)]
            in_channel = v
   return nn.Sequential(*layers)   # 将列表解开成独立的参数


class VGG16(nn.Module):
    def __init__(self, feature_extractor, num_classes=10):
        super(VGG16, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
        )
        self.init_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__ == '__main__':

    src = '../../study/data/cifar'
    model_path = '../../.cache/torch/checkpoints/vgg16-397923af.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    epoch = 1
    batch_size = 32
    lr = 0.001

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    train = datasets.CIFAR10(root=src, train = True,  transform=transform,  download=False)
    test = datasets.CIFAR10(root=src, train=False, transform=transform, download=False)

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False)

    vgg16_bn = VGG16(make_layers(cfg, batch_norm=True)).to(device)
    # vgg16_bn.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss().to(device)
    optim = optim.Adam(vgg16_bn.parameters(), lr=lr)

    vgg16_bn.train()
    for i in range(epoch):
        right_tr = 0
        for x_tr, y_tr in train_loader:
            x_tr = x_tr.to(device)
            y_tr = y_tr.to(device)

            out = vgg16_bn(x_tr)
            loss = criterion(out, y_tr)

            optim.zero_grad()
            loss.backward()
            optim.step()

            right_tr += (torch.max(out, dim=1)[1] == y_tr).sum().item()
        print(right_tr, len(train), right_tr/len(train))

    vgg16_bn.val()
    with torch.no_grad():
        right_te = 0
        for x_te, y_te in test_loader:
            x_te = x_te.to(device)
            y_te = y_te.to(device)
            out = vgg16_bn(x_te)
            right_te += (torch.max(out, dim=1)[1] == y_te).sum().item()
        print(right_te, len(test), right_te / len(test))


'''
https://blog.csdn.net/xckkcxxck/article/details/82379854
https://blog.csdn.net/qq_30159015/article/details/80801381

'''





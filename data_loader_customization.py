import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

'''source: https://zhuanlan.zhihu.com/p/28200166'''

class Mydata(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]  #self.x.size(0)



if __name__ == '__main__':
    x = torch.arange(10).view((5, 2))
    y = torch.arange(5)
    print(x)
    print(y)

    data = Mydata(x, y)
    data_loader = DataLoader(data,
                             batch_size=2,
                             shuffle=True,
                             num_workers=0
                             )
    for x, y in data_loader:
        print(x, y)
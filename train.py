import torch
from torch import optim, nn, utils
from torch._C import device
from torch.autograd import Variable
from netz import NetZ
from datasets import Datasets
import numpy as np
import random


torch.manual_seed(42)
random.seed(42)


def main():
    # ハイパーパラメータ
    batch_size = 128
    lr = 0.0005
    momentum = 0.9
    num_epochs = 10

    # データセット [train:val:test] = [64:16:20]
    dataset = Datasets('./X5000.npy', './y5000.npy')
    n_samples = len(dataset)
    index_list = list(range(n_samples))
    random.shuffle(index_list)
    train_index = index_list[:int(n_samples * 0.64)]
    val_index = index_list[int(n_samples * 0.64):int(n_samples * 0.8)]
    test_index = index_list[int(n_samples * 0.8):]

    train_dataset = utils.data.dataset.Subset(dataset, train_index)
    train_loader = utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = utils.data.dataset.Subset(dataset, val_index)
    val_loader = utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataset = utils.data.dataset.Subset(dataset, test_index)
    test_loader = utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデル
    model = NetZ().to(device)

    # 最適化手法
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 損失関数
    criterion = nn.MSELoss()

    # 学習
    for epoch in range(num_epochs):
        train(train_loader, model, device, optimizer, epoch, criterion)
        test(test_loader, model, device)


def train(train_loader, model, device, optimizer, epoch, criterion):
    model.train()  # 学習モードに切り替える

    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.reshape(-1)
        loss = criterion(outputs.float(), targets.float())
        loss.backward()
        optimizer.step()
        if idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f})'.format(
                epoch, idx * len(inputs), len(train_loader.dataset), 
                100. * idx / len(train_loader), loss.item()))

def test(test_loader, model, device):
    model.eval()  # 評価モードに切り替える
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.reshape(-1)
            criterion = nn.MSELoss(reduction='sum')
            test_loss += criterion(outputs.float(), targets.float()).item()
        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    main()
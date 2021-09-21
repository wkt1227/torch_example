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
    num_epochs = 3

    inputs_path = './data/X5000.npy'
    targets_path = './data/y5000.npy'
    best_model_path = './models/best_model.pth'

    # データセット [train:val:test] = [64:16:20]
    dataset = Datasets(inputs_path, targets_path)
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

    # 学習履歴
    history = {
        'loss': [],
        'val_loss': []
    }

    min_val_loss = {
        'value': float('inf'),
        'epoch': 0
    }

    # 学習
    for epoch in range(num_epochs):
        loss = train(train_loader, model, device, optimizer, criterion)
        val_loss = test(val_loader, model, device, criterion)
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)

        if min_val_loss['value'] > val_loss:
            min_val_loss['value'] = val_loss
            min_val_loss['epoch'] = epoch 
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch [{epoch+1}/{num_epochs}] - loss: {loss:.4f} - val_loss: {val_loss:.4f}')

    # 最小val_loss
    print(f'Min val_loss: {min_val_loss["value"]:.4f} - epoch: {min_val_loss["epoch"]+1}')

    # テスト
    test_loss = test(test_loader, model, device, criterion)
    print(f'Test - loss: {test_loss:.4f}')


def train(train_loader, model, device, optimizer, criterion):
    model.train()  # 学習モードに切り替える

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(inputs)
        outputs = outputs.reshape(-1)  # outputsとtargetsのshapeを統一する
        loss = criterion(outputs.float(), targets.float())
        loss.backward()  # 誤差を伝播、勾配を計算
        optimizer.step()  # 重みの更新
        
    return loss


def test(test_loader, model, device, criterion):
    model.eval()  # 評価モードに切り替える
    test_loss = 0
    with torch.no_grad():  # 計算グラフを構築しない（メモリ節約）
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.reshape(-1)  # outputsとtargetsのshapeを統一する
            test_loss += criterion(outputs.float(), targets.float()).item()
        
    test_loss /= len(test_loader)
    
    return test_loss

if __name__ == '__main__':
    main()
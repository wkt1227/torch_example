from datasets import Datasets
from torch import utils
import matplotlib.pyplot as plt


dataset = Datasets('./X5000.npy', './y5000.npy')
dataloader = utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for X, y in dataloader:
    print(X.shape)

img = X[0]
fig, ax = plt.subplots(1, 5)
for i in range(5):
    ax[i].imshow(img[i, :, :])

plt.show()
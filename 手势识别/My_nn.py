from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from DataLoader import My_Dataset

class My_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 126),
            nn.Linear(126, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.network(x)
        return x


def train(model, loss_fn, optimizer, dataloader, device, epoch=200, ):
    total_train_step = 0
    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i + 1))

        # 训练步骤开始
        model.train()
        for data in dataloader:
            X_train, y_train = data
            X_train = torch.stack(X_train).T.to(device)
            X_train = X_train.unsqueeze(1).float()
            y_train = y_train.unsqueeze(1).float()
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            predict = model(X_train)
            loss = loss_fn(predict, y_train)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
    torch.save(my_nn, "my_nn.pth")


def predict(model, input):
    predict_output = model(input)
    print(predict_output)


if __name__ == "__main__":
    device = torch.device("cuda")

    train_dataset = My_Dataset("final.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0)

    my_nn = My_nn()
    my_nn = my_nn.to(device)

    loss_fn = nn.L1Loss()
    loss_fn = loss_fn.to(device)
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=learning_rate)

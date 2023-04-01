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
        )

    def forward(self, x):
        x = self.network(x)
        return x


class Model:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = My_nn().to(self.device)
        self.loss_fn = nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def train(self):
        train_dataset = My_Dataset("final.csv")
        train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True, num_workers=0)
        total_train_step = 0
        epoch = 200
        for i in range(epoch):
            print("-------第 {} 轮训练开始-------".format(i + 1))

            # 训练步骤开始
            self.model.train()
            for data in train_dataloader:
                X_train, y_train = data
                X_train = torch.stack(X_train).T.to(self.device)
                X_train = X_train.float()
                y_train = y_train.unsqueeze(1).float()
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                predict = self.model(X_train)
                loss = self.loss_fn(predict, y_train)

                # 优化器优化模型
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_step += 1
                if total_train_step % 100 == 0:
                    print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        torch.save(self.model, "my_nn.pth")


if __name__ == "__main__":
    model = Model()
    model.train()


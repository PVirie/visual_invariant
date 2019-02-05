import torch
import numpy as np


class Nearest_Neighbor:

    def __init__(self, device, file_path=None):
        print("init")
        self.device = device
        self.weights = []
        self.file_path = file_path

    def save(self):
        if self.file_path:
            torch.save(self.weights, self.file_path)

    def load(self):
        if self.file_path:
            self.weights = torch.load(self.file_path)

    def learn(self, input, output, num_classes):
        print("learn")

        # expand
        new_weight = (torch.transpose(input, 0, 1), output)

        # merge
        self.weights.append(new_weight)

    def __internal__forward(self, input, weights):

        logits = torch.sum(input * input, dim=1, keepdim=True) - torch.cat([
            - 2 * torch.matmul(input[:, :A.shape[0]], A) + torch.sum(A * A, dim=0, keepdim=True)
            for (A, B) in weights
        ], dim=1)

        return logits

    # ----------- public functions ---------------

    def logit(self, input, depth_out):

        with torch.no_grad():
            predictions = self.__lshift__(input)
            logits_ = torch.zeros([input.shape[0], depth_out], device=self.device)
            logits_[np.arange(input.shape[0]), predictions] = 1.0

        return logits_

    def __lshift__(self, input):
        with torch.no_grad():
            logits_ = self.__internal__forward(input, self.weights)

            indices = torch.argmax(logits_, dim=1)

            bases = torch.cat([
                B for (A, B) in self.weights
            ], dim=0)

            prediction = bases[indices]

        return prediction


if __name__ == '__main__':
    print("test nearest neighbor")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Nearest_Neighbor(device)

    x = torch.randn(100, 392, device=device)
    y = torch.randint(5, (100, ), dtype=torch.int64, device=device)

    layer.learn(x, y, num_classes=5)

    y_ = layer << x
    print(y)
    print(y_)
    print("Percent correct: ", torch.sum(y_ == y).item() * 100 / x.shape[0])

    x2 = torch.randn(100, 784, device=device)
    y2 = torch.randint(10, (100, ), dtype=torch.int64, device=device)

    layer.learn(x2, y2, num_classes=10)

    x3 = torch.randn(100, 784, device=device)
    y3 = torch.randint(10, (100, ), dtype=torch.int64, device=device)

    layer.learn(x3, y3, num_classes=10)

    xs = torch.zeros(x.shape[0], x2.shape[1], device=device)
    xs[:, 0:x.shape[1], ...] = x
    y_ = layer << xs
    print(y_)
    print("Percent correct: ", torch.sum(y_ == y).item() * 100 / x.shape[0])

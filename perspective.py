import torch
import cv2
import numpy as np
import math
from dataset import FashionMNIST


def transform(x, affine):
    theta = affine.unsqueeze(0).expand(x.shape[0], -1, -1)
    grid = torch.nn.functional.affine_grid(theta, x.size())
    output = torch.nn.functional.grid_sample(x, grid)
    return output


def perspective(x, affines):
    b = x.shape[0]
    p = affines.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    thetas = torch.reshape(affines.unsqueeze(0).expand(b, -1, -1, -1), [-1, 2, 3])
    cross = torch.reshape(x.unsqueeze(1).expand(-1, p, -1, -1, -1), [-1, c, h, w])

    grid = torch.nn.functional.affine_grid(thetas, [b * p, c, h, w])
    output = torch.reshape(torch.nn.functional.grid_sample(cross, grid), [b, p, c, h, w])
    return output


def get_sample_kernels(scope):
    ts = np.linspace(scope[0][0], scope[0][1], scope[0][2])
    ys = np.linspace(scope[1][0], scope[1][1], scope[1][2])
    xs = np.linspace(scope[2][0], scope[2][1], scope[2][2])

    C = np.tile(np.reshape(np.cos(ts), [ts.shape[0], 1, 1]), [1, scope[1][2], scope[2][2]]).flatten()
    S = np.tile(np.reshape(np.sin(ts), [ts.shape[0], 1, 1]), [1, scope[1][2], scope[2][2]]).flatten()

    Y = np.tile(np.reshape(ys, [1, ys.shape[0], 1]), [scope[0][2], 1, scope[2][2]]).flatten()
    X = np.tile(np.reshape(xs, [1, 1, xs.shape[0]]), [scope[0][2], scope[1][2], 1]).flatten()

    Xt = C * X - S * Y
    Yt = S * X + C * Y

    row0 = np.stack([C, -S, Xt], axis=1)
    row1 = np.stack([S, C, Yt], axis=1)

    thetas = np.stack([row0, row1], axis=1)

    return thetas


if __name__ == '__main__':
    print("assert perspective.")

    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FashionMNIST(device, batch_size=10, max_per_class=2, seed=100, group_size=1)

    thetas = get_sample_kernels([[-math.pi / 6, math.pi / 6, 3], [-0.3, 0.3, 5], [-0.3, 0.3, 5]])
    affines = torch.tensor(thetas, dtype=torch.float32, device=device)

    for i, (data, label) in enumerate(dataset):
        print(data.shape, label.shape)
        print(dataset.readout(label))

        input = data.to(device)
        output = label.to(device)

        gen = perspective(input, affines)
        gen_cpu = gen.cpu().numpy()
        print(gen_cpu.shape)

        img = np.concatenate([
            1.0 - np.reshape(data.numpy()[:, 0, ...], [-1, 28]),
            np.reshape(np.transpose(gen_cpu[:, :, 0, ...], [0, 2, 1, 3]), [-1, 28 * len(affines)])
        ], axis=1)

        cv2.imshow("sample", img)
        cv2.waitKey(-1)

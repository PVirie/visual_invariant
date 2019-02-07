import torch
import numpy as np
import cv2
from dataset import FashionMNIST
from linear import Conceptor
from semantic import Semantic_Memory
from nearest import Nearest_Neighbor
from perspective import *


class Static_Hierarchy_Classifier:
    def __init__(self, device, num_classes):
        print("init")
        self.device = device
        self.num_classes = num_classes

        # width, height
        self.sample_size = (9, 9)
        self.perspective_dim = (1, 19, 19)
        self.perspective_count = self.perspective_dim[0] * self.perspective_dim[1] * self.perspective_dim[2]

        self.part = {}
        self.models = {}
        self.view_param = {}

        self.part[0] = get_perspective_kernels([[0, 0, 1], [0, 0, 1], [0, 0, 1]], scale=1)
        self.view_param[0] = get_perspective_kernels([[0, 0, self.perspective_dim[0]], [-0.2, 0.2, self.perspective_dim[1]], [0.2, 0.2, self.perspective_dim[2]]], scale=1)

        self.part[1] = get_perspective_kernels([[0, 0, 1], [-0.5, 0.5, 2], [-0.5, 0.5, 2]], scale=2)
        self.view_param[1] = get_perspective_kernels([[0, 0, self.perspective_dim[0]], [-0.1, 0.1, self.perspective_dim[1]], [0.1, 0.1, self.perspective_dim[2]]], scale=1)

        self.part[2] = get_perspective_kernels([[0, 0, 1], [-0.2, 0.2, 3], [-0.2, 0.2, 3]], scale=2)
        self.view_param[2] = get_perspective_kernels([[0, 0, self.perspective_dim[0]], [-0.05, 0.05, self.perspective_dim[1]], [0.05, 0.05, self.perspective_dim[2]]], scale=1)

        self.semantic = Nearest_Neighbor(device)

        self.running_base_position = 0

    def to_tensor(self, input, dtype=torch.float32):
        return torch.tensor(input, dtype=dtype, device=self.device)

    def get_min_index(self, score):
        self.mid = self.perspective_count // 2
        score[:, self.mid] -= 1e-6
        index = torch.argmin(score, dim=1, keepdim=True)
        return index

    def layer(self, input, layer, id="", base_perspective=None, and_learn=False):
        batches = input.shape[0]

        bases = []
        orders = []

        if base_perspective is not None:
            part_perspective = np.reshape(rebase(self.part[layer], base_perspective), [-1, 2, 3])  # (batch*num part, 2, 3)
        else:
            part_perspective = np.tile(self.part[layer], (input.shape[0], 1, 1))

        count_parts = self.part[layer].shape[0]
        count_views = self.view_param[layer].shape[0]

        part_perspective = rebase(self.view_param[layer], part_perspective)  # (batch*num part, num perspective, 2, 3)
        perspectives = sample(input, self.to_tensor(np.reshape(part_perspective, [batches, -1, 2, 3])), size=self.sample_size)  # (batch, num part * num perspective, ...)
        perspectives = torch.reshape(perspectives, [batches, count_parts, count_views, -1])
        for i in range(count_parts):
            _flat = torch.reshape(perspectives[:, i, ...], [batches * count_views, -1])

            _id = id + str(i)
            if _id not in self.models:
                self.models[_id] = Conceptor(self.device, max_bases=10)

            if self.models[_id].get_count() == 0:
                _projected = _flat
            else:
                _projected = self.models[_id].project(_flat)

            scores = torch.mean(torch.reshape((_flat - _projected)**2, [batches, count_views, -1]), dim=2)
            min_index = self.get_min_index(scores)
            min_indices = torch.unsqueeze(min_index, 2).repeat(1, 1, perspectives.shape[3])
            min_perspective = torch.gather(perspectives[:, i, ...], 1, min_indices)[:, 0, ...]

            if and_learn:
                self.running_base_position += self.models[_id].learn(min_perspective, 1, start_base_order=self.running_base_position, expand_threshold=1e-3)

            hidden = (self.models[_id]) << min_perspective
            bases.append(hidden)
            orders.extend(self.models[_id].get_orders())

            if layer < len(self.part) - 1:
                min_view_param = self.view_param[layer][np.squeeze(min_index.cpu().numpy()), ...]
                if len(min_view_param.shape) < 3:
                    min_view_param = np.expand_dims(min_view_param, 0)
                _b, _o = self.layer(input, layer + 1, _id, min_view_param, and_learn)
                bases += _b
                orders += _o

        return bases, orders

    def reorder_bases(self, bases, orders):
        _b = torch.cat(bases, dim=1)
        return _b[:, orders]

    def classify(self, input, output=None):

        and_learn = output is not None

        mark = self.running_base_position
        bases, orders = self.layer(input, 0, "", None, and_learn=and_learn)
        logits = self.reorder_bases(bases, orders)

        if mark == 0:
            prediction = torch.randint(10, (input.shape[0], ), dtype=torch.int64, device=device)
        else:
            prediction = self.semantic << logits[:, 0:mark]

        if and_learn:
            self.semantic.learn(logits, output, num_classes=self.num_classes)

        return prediction


if __name__ == "__main__":
    print("main")

    device = torch.device("cuda:0")

    batch_size = 1
    dataset = FashionMNIST(device, batch_size=batch_size, max_per_class=60, seed=10, group_size=2)

    classifier = Static_Hierarchy_Classifier(device, 10)

    percent_correct = 0.0
    for i, (data, label) in enumerate(dataset):
        print("data: ", i)

        input = data.to(device)
        output = label.to(device)

        # online test
        current_bits = 0
        prediction = classifier.classify(input, output=output).cpu()
        count_correct = np.sum(prediction.numpy() == label.numpy())
        percent_correct = 0.99 * percent_correct + 0.01 * count_correct * 100 / batch_size
        print("Truth: ", dataset.readout(label))
        print("Guess: ", dataset.readout(prediction))
        print("Percent correct: ", percent_correct)

        img = np.reshape(data.numpy(), [-1, data.shape[2]])
        cv2.imshow("sample", img)
        cv2.waitKey(10)

    count = 0
    for i, (data, label) in enumerate(dataset):
        input = data.to(device)
        output = label.to(device)

        # test
        prediction = classifier.classify(input).cpu()
        count = count + np.sum(prediction.numpy() == label.numpy())

    print("Percent correct: ", count * 100 / (len(dataset) * batch_size))

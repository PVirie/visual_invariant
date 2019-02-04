import torch


class Semantic_Memory:
    # unlike nearest neighbor, semantic memory always requires that new difference are given as new dimensions.
    # this is like the original conceptor in a way.

    def __init__(self, device, file_path=None):
        print("init")
        self.device = device
        self.weights = []
        self.new_weights = []
        self.current_depth = 0
        self.file_path = file_path

    def save(self):
        if self.file_path:
            torch.save(self.weights, self.file_path)

    def load(self):
        if self.file_path:
            self.weights = torch.load(self.file_path)

    def learn(self, input, output, num_classes, expand_threshold=1e-2, steps=1000, lr=0.01, verbose=False):
        print("learn")

        with torch.no_grad():
            if len(self.weights) is not 0:
                prev_logits_ = self.__internal__forward(input, self.weights, num_classes)
            else:
                prev_logits_ = torch.zeros(input.shape[0], num_classes, device=self.device)

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # expand
        A = torch.empty(input.shape[1] - self.current_depth, num_classes, device=self.device, requires_grad=True)
        torch.nn.init.normal_(A, 0, 0.001)
        self.new_weights.append(A)

        expanded_input = input[:, self.current_depth:]

        optimizer = torch.optim.Adam(self.new_weights, lr=lr)
        for i in range(steps):

            logits_ = self.__internal__forward(expanded_input, self.new_weights)

            loss = criterion(prev_logits_ + logits_, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                if verbose:
                    print("step:", i, "th, loss:", loss.item())

        if verbose:
            print("final loss:", loss.item())

        # merge
        self.weights.append(A)
        self.new_weights.clear()
        self.current_depth = input.shape[1]

    def __internal__forward(self, input, weights, depth_out=0):

        for f in weights:
            depth_out = max(depth_out, f.shape[1])

        canvas = torch.zeros([input.shape[0], depth_out], device=self.device)

        from_depth = 0
        for f in weights:
            to_depth = from_depth + f.shape[0]
            addition = torch.matmul(input[:, from_depth:to_depth, ...], f)
            occupied_depth = f.shape[1]
            canvas[:, 0:occupied_depth, ...] = canvas[:, 0:occupied_depth, ...] + addition
            from_depth = to_depth

        return canvas

    # ----------- public functions ---------------

    def logit(self, input, depth_out):
        with torch.no_grad():
            logits_ = self.__internal__forward(input, self.weights, depth_out)

        return logits_

    def __lshift__(self, input):
        with torch.no_grad():
            logits_ = self.__internal__forward(input, self.weights)

            prediction = torch.argmax(logits_, dim=1)

        return prediction


if __name__ == '__main__':
    print("test semantic memory")

    dtype = torch.float
    device = torch.device("cuda:0")

    layer = Semantic_Memory(device)

    x = torch.randn(10, 5, device=device)
    y = torch.randint(5, (10, ), dtype=torch.int64, device=device)

    layer.learn(x, y, num_classes=5, verbose=True)

    y_ = layer << x
    print(y)
    print(y_)
    print("Percent correct: ", torch.sum(y_ == y).item() * 100 / x.shape[0])

    x2 = torch.randn(10, 10, device=device)
    y2 = torch.randint(10, (10, ), dtype=torch.int64, device=device)

    layer.learn(x2, y2, num_classes=10, verbose=True)

    x3 = torch.randn(10, 15, device=device)
    y3 = torch.randint(15, (10, ), dtype=torch.int64, device=device)

    layer.learn(x3, y3, num_classes=15, verbose=True)

    xs = torch.zeros(x.shape[0], x3.shape[1], device=device)
    xs[:, 0:x.shape[1], ...] = x
    y_ = layer << xs

    print(y_)
    print("Percent correct: ", torch.sum(y_ == y).item() * 100 / x.shape[0])

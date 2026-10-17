import torch
from torch.func import vmap


class FitnessFunction:
    def __init__(self, model, dataloader, chunk_size, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.chunk_size = chunk_size

        self.model.eval()


    def __call__(self, parameters):
        parameters = parameters.to(self.device)
        correct = torch.zeros((parameters.shape[0],), device=self.device)

        total = 0
        with torch.no_grad():
            parallel_eval_fn = vmap(self.model.parallel_eval, in_dims=(0, None), chunk_size=self.chunk_size)
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)

                outputs = parallel_eval_fn(parameters, data.to(self.device))
                _, predicted = torch.max(outputs, 2)

                correct += torch.sum(target.to(self.device) == predicted, 1)
                total += target.shape[0]

        return correct / total
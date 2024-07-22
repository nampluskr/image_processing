import numpy as np
import torch

import sys
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = {"loss": loss_fn}
        self.metrics.update(metrics)

        self.options = dict(leave=False, file=sys.stdout, ascii=True, ncols=100)
        self.device = next(model.parameters()).device

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()

        res = {"loss": loss.item()}
        for metric, metric_fn in self.metrics.items():
            if metric != "loss":
                res[metric] = metric_fn(pred, y).item()
        return res

    @torch.no_grad()
    def test_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        res = {metric: metric_fn(pred, y).item() 
               for metric, metric_fn in self.metrics.items()}
        return res

    def update_history(self, res):
        for metric, value in res.items():
            self.history[metric].append(value.mean())

    def fit(self, train_loader, n_epochs, valid_loader=None):
        self.history = {metric: [] for metric in self.metrics}
        if valid_loader is not None:
            self.history.update({f"val_{metric}": [] for metric in self.metrics})

        for i in range(1, n_epochs + 1):
            epoch = str(i).rjust(len(str(n_epochs)), ' ')
            epoch = f"Epoch[{epoch}/{n_epochs}] "

            ## Training
            self.model.train()
            res = {metric: np.array([]) for metric in self.metrics}
            with tqdm(train_loader, **self.options) as pbar:
                for x, y in pbar:
                    step_res = self.train_step(x, y)
                    for metric in step_res:
                        res[metric] = np.append(res[metric], step_res[metric])

                    train_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in res.items()])
                    pbar.set_description(">> " + epoch + train_desc)

            ## Validation
            self.model.eval()
            if valid_loader is None:
                print(">> " + epoch + train_desc)
                self.update_history(res)
                continue

            val_res = {f"val_{metric}": np.array([]) for metric in self.metrics}
            with tqdm(valid_loader, **self.options) as pbar:
                for x, y in pbar:
                    step_res = self.test_step(x, y)
                    for metric in step_res:
                        val_res[f"val_{metric}"] = np.append(val_res[f"val_{metric}"],
                                                             step_res[metric])
                    valid_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in val_res.items()])
                    pbar.set_description(">> " + epoch + valid_desc)

            print(">> " + epoch + train_desc, "|", valid_desc)
            self.update_history(res)
            self.update_history(val_res)

        return self.history

    def evaluate(self, test_loader):
        self.model.eval()
        test_res = {metric: np.array([]) for metric in self.metrics}
        with tqdm(test_loader, **self.options) as pbar:
            for x, y in pbar:
                step_res = self.test_step(x, y)
                for metric in test_res:
                    test_res[metric] = np.append(test_res[metric], step_res[metric])

                test_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in test_res.items()])
                pbar.set_description(">> " + test_desc)

        return {metric: value.mean() for metric, value in test_res.items()}


if __name__ == "__main__":

    pass

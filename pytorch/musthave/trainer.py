import numpy as np
import torch

import sys
# from tqdm.auto import tqdm
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):
        self.counter += 1
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif  self.patience <= self.counter:
            return True
        return False


class Trainer:
    def __init__(self, model, optimizer, loss_fn, metrics={}):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = {"loss": loss_fn}
        self.metrics.update(metrics)

        self.kwargs = dict(leave=False, file=sys.stdout, unit=" batch",
                           ascii=True, ncols=100)
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

    def fit(self, train_loader, n_epochs, valid_loader=None,
            lr_scheduler=None, early_stopper=None):
        self.history = {metric: [] for metric in self.metrics}
        if valid_loader is not None:
            self.history.update({f"val_{metric}": [] for metric in self.metrics})

        for i in range(1, n_epochs + 1):
            epoch = str(i).rjust(len(str(n_epochs)), ' ')
            epoch = f"Epoch[{epoch}/{n_epochs}] "

            ## Training
            self.model.train()
            res = {metric: np.array([]) for metric in self.metrics}
            with tqdm(train_loader, **self.kwargs) as pbar:
                for x, y in pbar:
                    step_res = self.train_step(x, y)
                    for metric in step_res:
                        res[metric] = np.append(res[metric], step_res[metric])

                    train_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in res.items()])
                    pbar.set_description(">> " + epoch + train_desc)

            if lr_scheduler is not None:
                lr_scheduler.step()

            if valid_loader is None:
                print(">> " + epoch + train_desc)
                self.update_history(res)
                continue

            ## Validation
            self.model.eval()
            val_res = {f"val_{metric}": np.array([]) for metric in self.metrics}
            with tqdm(valid_loader, **self.kwargs) as pbar:
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

            ## Early Stopping
            if early_stopper is not None:
                val_loss = val_res["val_loss"].mean()
                if early_stopper.step(val_loss):
                    print(">> Early stopped!\n") # log
                    break

        return self.history

    def evaluate(self, test_loader):
        self.model.eval()
        test_res = {metric: np.array([]) for metric in self.metrics}
        with tqdm(test_loader, **self.kwargs) as pbar:
            for x, y in pbar:
                step_res = self.test_step(x, y)
                for metric in test_res:
                    test_res[metric] = np.append(test_res[metric], step_res[metric])

                test_desc = ', '.join([f"{m}={v.mean():.3f}" for m, v in test_res.items()])
                pbar.set_description(">> " + test_desc)

        return {metric: value.mean() for metric, value in test_res.items()}


if __name__ == "__main__":

    pass

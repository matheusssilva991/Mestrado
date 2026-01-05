#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn.functional as F

dtype = torch.float


class TwoLayerNet(torch.nn.Module):
    '''
        From the pytorch examples, a simjple 2-layer neural net.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    def __init__(self, d_in, hidden_size, d_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def linear_model(x, y, epochs=200, hidden_size=10):
    '''
        Predict y from x using a simple linear model with one hidden layer.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    assert x.shape[0] == y.shape[0], 'x and y have different batch sizes'
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = TwoLayerNet(d_in, hidden_size, d_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    errors = []
    for t in range(epochs):
        y_pred = model(x)
        tot_loss = criterion(y_pred, y)
        # No linear_model:
        denom = y.sum().item()
        if denom == 0:
            perc_loss = 0
        else:
            perc_loss = 100. * torch.sqrt(tot_loss).item() / denom
        errors.append(perc_loss)
        if t % 10 == 0 or epochs < 20:
            print('epoch {:4d}: {:.5f} {:.2f}%'.format(t, tot_loss, perc_loss))
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    return model, errors


def plotErrors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Percentage error')
    plt.xlabel('Epoch')
    plt.show()


def plotResults(y_actual, y_predicted):
    '''
        Plot the actual and predicted y values (in different colours).
    '''
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _plot_mfs(var_name, fv, x):
    '''
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    '''
    # Sort x so we only plot each x-value once:
    xsort, _ = x.sort()
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
    plt.ylabel('Membership')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()


def plot_all_mfs(model, x):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                               / y_actual))
    return(tot_loss, rmse, perc_loss)


def test_anfis(model, data, show_plots=False):
    '''
        Do a single forward pass with x and compare with y_actual.
    '''
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print('### Testing for {} cases'.format(x.shape[0]))
    y_pred = model(x)
    mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rmse, perc_loss))
    if show_plots:
        plotResults(y_actual, y_pred)


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False, val_loader=None, early_stop=20):
    errors = []
    val_accs = []
    best_acc = 0
    best_state = None
    best_epoch = 0
    no_improve = 0

    print('### Training for {} epochs, training size = {} cases'.format(epochs, data.dataset.tensors[0].shape[0]))

    for t in trange(epochs, desc="Treinando"):
        # Treino
        model.train()
        for x_batch, y_batch in data:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Ajustar coeficientes (LSE) se híbrido
        if model.hybrid:
            x, y_actual = data.dataset.tensors
            with torch.no_grad():
                model.fit_coeff(x, y_actual)

        # Calcular erro de treino
        model.eval()
        with torch.no_grad():
            x, y_actual = data.dataset.tensors
            y_pred = model(x)
            mse, rmse, perc_loss = calc_error(y_pred, y_actual)
            errors.append(perc_loss)

        # Validação e early stopping
        if val_loader is not None:
            x_val, y_val = val_loader.dataset.tensors
            with torch.no_grad():
                y_val_pred = model(x_val)
                cat_pred = torch.argmax(y_val_pred, dim=1)
                cat_act = torch.argmax(y_val, dim=1)
                val_acc = (cat_pred == cat_act).sum().item() / len(cat_act)
                val_accs.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict().copy()
                best_epoch = t
                no_improve = 0
            else:
                no_improve += 1

            if t % 5 == 0:
                print(f"Época {t}: MSE={mse:.5f}, RMSE={rmse:.5f}, Val Acurácia={val_acc:.4f}")

            if no_improve >= early_stop:
                print(f"Early stopping ativado na época {t}!")
                break
        else:
            if t % 5 == 0:
                print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%'.format(t, mse, rmse, perc_loss))

    # Restaurar melhor modelo
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Melhor modelo restaurado da época {best_epoch} (Val Acurácia={best_acc:.4f})")

    if show_plots:
        plotErrors(errors)
        if val_loader is not None:
            plt.plot(val_accs, label='Validação')
            plt.xlabel('Época')
            plt.ylabel('Acurácia')
            plt.legend()
            plt.show()
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plotResults(y_actual, y_pred)


def train_anfis(model, data, epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots)


if __name__ == '__main__':
    x = torch.arange(1, 100, dtype=dtype).unsqueeze(1)
    y = torch.pow(x, 3)
    model, errors = linear_model(x, y, 100)
    plotErrors(errors)
    plotResults(y, model(x))

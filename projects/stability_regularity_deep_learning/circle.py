import torch

from datascience.data.synthetize.create_dataset import create_dataset
from datascience.ml.neural.supervised.callbacks.circle import CircleCallback
from datascience.ml.neural.supervised import fit
from datascience.ml.neural.models import FullyConnected

# creating dataset
from datascience.ml.neural.loss.loss import CELoss
import numpy as np

from datascience.ml.neural.checkpoints.checkpoints import create_model
from datascience.visu.deep_test_plots import plot_decision_boundary, plot_dataset, plot_gradient_field, \
    plot_activation_rate
from datascience.visu.util.util import plt, save_fig
from engine.parameters import special_parameters

train, test = create_dataset(param_train=(250, 250, True, {'scale': 0.42}), poly=False)
# creating/loading a model
model_params = {
    'architecture': (1,),
    'dropout': 0.0,
    'batchnorm': True,
    'bias': True,
    'relu': True
}

model = create_model(model_class=FullyConnected, model_params=model_params)

for k, v in model.named_parameters():
    print(k, v)
# exit()
training_params = {
    'lr': 0.1,
    'iterations': [200],
    'log_modulo': -1,
    'val_modulo': 1,
}
validation_params = {
    'vcallback': (CircleCallback(bias=True, wk=False),),
    # 'metrics': (ValidationAccuracy(1),)
}
optim_params = {
    'momentum': 0.0
}

fit(
    model, train=train, test=test, training_params=training_params,
    validation_params=validation_params, optim_params=optim_params
)

if hasattr(special_parameters, 'circle_all'):
    #######################
    # out of modules tries
    #######################
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=5, num_workers=16)

    data, labels = next(iter(train_loader))
    model.hidden_layer = True
    result = model(data)
    loss = CELoss()
    loss_value = loss(result, labels)
    loss_value.backward()

    r = np.linspace(-np.pi, np.pi, 2000)
    data = torch.from_numpy(np.array([(np.cos(i), np.sin(i)) for i in r])).float()

    labels = torch.from_numpy(np.array([1 if i < 1000 else 0 for i in range(2000)])).long()
    train_bis = torch.utils.data.TensorDataset(data, labels)
    train_loader = torch.utils.data.DataLoader(train_bis, shuffle=False, batch_size=1, num_workers=16)

    loss_values = []
    model.eval()
    model.hidden_layer = False
    for i, (data, label) in enumerate(train_loader):
        result = model(data)

        loss_values.append(label.detach().numpy()[0]-result[0][1].detach().numpy())  # *norm.pdf(r[i], np.pi/2, 0.5))
    plt('gaussian').plot(np.linspace(-np.pi, np.pi, 2000), loss_values)

    #######################
    # plots inside modules
    #######################
    ax = plot_dataset(train.dataset, train.labels)
    plot_activation_rate(train.dataset, train.labels, model, ax=ax)

    plot_decision_boundary(model, ax=ax)

    plot_gradient_field(train.dataset, train.labels, model, ax=ax)

    save_fig()

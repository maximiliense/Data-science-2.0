from datascience.visu.deep_test_plots import DeriveNeuronOutput
from datascience.ml.neural.loss.loss import CELoss
from engine.parameters import special_parameters
from engine.core import module

import torch
from torch.autograd import Variable
from torch.autograd import grad


@module
def second_derivative(model, train, loss=CELoss(), layer=-1, neuron=-1):
    print("Second derivative:")
    num_workers = special_parameters.nb_workers
    test_loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=8, num_workers=num_workers)

    all_outputs = []
    all_labels = []

    for batch in test_loader:
        data, labels = batch
        output = model(data)

        all_labels.append(labels)
        all_outputs.append(output)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    gradients = grad(loss.loss(all_outputs, all_labels), model.parameters(), create_graph=True)

    jac = torch.cat([r.flatten() for r in gradients])

    hess = []
    for j in jac:
        model.zero_grad()
        j.backward(retain_graph=True)
        hess.append(torch.cat([p.grad.flatten() for p in model.parameters()]))

    hessian = torch.stack(hess).numpy()
    return hessian


@module
def jacobian(model, train, loss=CELoss(), layer=-1, neuron=-1, parameters=None):
    print("\n\n*******\n\nJacobian:")
    num_workers = special_parameters.nb_workers
    test_loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=8, num_workers=num_workers)

    all_outputs = []
    all_labels = []

    for batch in test_loader:
        data, labels = batch
        output = model(data)

        all_labels.append(labels)
        all_outputs.append(output)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    gradients = grad(loss.loss(all_outputs, all_labels), model.parameters(), create_graph=True)


    jac = torch.cat([r.flatten() for r in gradients])

    JJT = torch.ger(jac,jac).detach().numpy()


    return JJT


@module
def neuron_derivative(model, X):
    print('Hello')
    for param in model.parameters():

        param.requires_grad = False
    hook = DeriveNeuronOutput(layer=4)
    hook.register(model)

    data = Variable(torch.from_numpy(X).float(), requires_grad=True)

    _ = model(data)
    print(data.grad)
    hook.remove()

    # results = hook.finalize()

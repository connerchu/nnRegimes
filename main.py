from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call, vmap, jacrev
import matplotlib.pyplot as plt
import argparse

# set precision to float64 to prevent numerical overflow in kernel calculation
torch.set_default_dtype(torch.float64)

def parse_args():
    parser = argparse.ArgumentParser(description="NTK vs Narrow/Wide Width NN")
    parser.add_argument("--width", type=int, default=100, help="Width of NN")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of Epochs")
    parser.add_argument("--save_name", type=str, default="test.png", help="Name of the output plot file")
    return parser.parse_args()

def true_function(x):
    return torch.sin(x)

# training set of 10 points on sine wave. test set of 100 points.
X_train = torch.linspace(-torch.pi,torch.pi,10).reshape(-1,1)
X_test = torch.linspace(-torch.pi,torch.pi,100).reshape(-1,1)
Y_train = true_function(X_train)
Y_test = true_function(X_test)

def make_model(width):
    return nn.Sequential(nn.Linear(1,width), nn.ReLU(), nn.Linear(width,1))

args = parse_args()

# initialize models
NTKmodel = make_model(args.width)
NTKparams = dict(NTKmodel.named_parameters())
sgd_model = make_model(args.width)
sgd_model.load_state_dict(NTKmodel.state_dict()) # initialize with same parameters as NTK model

def fnet_single(params,x): # run the model once for x using params
    return functional_call(NTKmodel,params,(x.unsqueeze(0),)).squeeze(0) # function_call expects x.shape: [1,1] so unsqueeze

# jacrev takes gradient of model wrt first arg i.e. every param and returns as dictionary
get_grads = vmap(jacrev(fnet_single),(None,0)) # vmap applies jacrev repeatedly with the same params but for each x

def buildNTKmatrix(x1,x2,params):
    grads1 = get_grads(params,x1) # returns python dict with key: name of parameter, value: size [N1,P]
    grads2 = get_grads(params,x2)
    # each row of g1_flat contains the gradient of f(x[i]) wrt all parameters flattened to one vector
    g1_flat = torch.cat([g.flatten(start_dim=1) for g in grads1.values()], dim=1)
    g2_flat = torch.cat([g.flatten(start_dim=1) for g in grads2.values()], dim=1)
    return torch.matmul(g1_flat, g2_flat.T) # builds the NTK gram matrix

K_train_train = buildNTKmatrix(X_train, X_train, NTKparams)
K_test_train = buildNTKmatrix(X_test, X_train, NTKparams)

# calculate initial output from random initialization so KRR only calculates the delta
with torch.no_grad():
    y_train_0 = NTKmodel(X_train)
    y_test_0 = NTKmodel(X_test)

# kernel regression solves (Ktrain + reg term)alpha=(ytrain-yinitial)
alpha = torch.linalg.solve(K_train_train + ((1e-4) * torch.eye(K_train_train.shape[0])), Y_train - y_train_0)
Y_NTKpred = y_test_0 + torch.matmul(K_test_train, alpha) # add f0 offset back to prediction

# adjust learning rate based on width to prevent gradient explosion
# heuristic: 0.5 / width keeps training stable in the lazy regime
learning_rate = 0.5 / args.width

# this isn't really SGD because we are using the whole batch but we would use SGD for larger datasets
sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=learning_rate)
sgd_loss = nn.MSELoss()

for epoch in range(args.epochs):
    sgd_optimizer.zero_grad() # clear the old gradients from memory
    sgd_pred = sgd_model(X_train) # predict outputs
    sgd_loss_value = sgd_loss(sgd_pred, Y_train) # calculate the loss for each data point
    sgd_loss_value.backward() # do back propagation (which uses the loss values)
    sgd_optimizer.step() # step in the -grad direction

with torch.no_grad():
    sgd_pred = sgd_model(X_test)

# calculate NTK post training
K_train_train_final = buildNTKmatrix(X_train, X_train, dict(sgd_model.named_parameters()))
# compare post and pre training NTKs
relative_change = torch.norm(K_train_train_final - K_train_train) / torch.norm(K_train_train)

# calculate RMSE accuracy on test set
ntk_rmse = torch.sqrt(torch.mean((Y_NTKpred - Y_test)**2)).item()
sgd_rmse = torch.sqrt(torch.mean((sgd_pred - Y_test)**2)).item()

plt.scatter(X_train.detach().numpy(), Y_train.detach().numpy())
plt.plot(X_test.detach().numpy(), Y_test.detach().numpy(), label='True Function')
plt.plot(X_test.detach().numpy(), Y_NTKpred.detach().numpy(), label=f'NTK Prediction (RMSE: {ntk_rmse:.4f})')
plt.plot(X_test.detach().numpy(), sgd_pred.detach().numpy(), label=f'SGD Prediction (RMSE: {sgd_rmse:.4f})', linestyle='--')

plt.title(f"NTK vs SGD Prediction: Width {args.width}, Epochs {args.epochs}\nRelative Kernel Change {relative_change.item():.5f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(args.save_name)
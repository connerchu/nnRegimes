import math
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

SEED = 0
N_TRAIN = 30
N_TEST = 200
X_MIN_TRAIN, X_MAX_TRAIN = -math.pi, math.pi
X_MIN_TEST, X_MAX_TEST = -3.5, 3.5
WIDTH_LIST = [32, 64, 128]
BASE_LR = 0.1 # lr = BASE_LR / width
TARGET_TIME = 40.0 # effective learning time = lr*steps for every width
RIDGE = 1e-6

def true_function(x):
    return torch.sin(3 * x)

torch.manual_seed(SEED)
X_train = (X_MIN_TRAIN + (X_MAX_TRAIN - X_MIN_TRAIN) * torch.rand(N_TRAIN, 1, device=device, dtype=dtype))
X_test  = torch.linspace(X_MIN_TEST,  X_MAX_TEST,  N_TEST,  device=device, dtype=dtype).reshape(-1, 1)

y_train_clean = true_function(X_train)
y_test_true = true_function(X_test).reshape(-1, 1)

y_train = y_train_clean.reshape(-1, 1)
X_train_np = X_train.detach().cpu().numpy()
y_train_np = y_train.detach().cpu().numpy()

# only evaluate RMSE min(X_train) <= x <= max(X_train) to avoid edges
x_lo = X_train.min()
x_hi = X_train.max()
domain_mask = (X_test >= x_lo) & (X_test <= x_hi)

class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)

def flatten_pytree(d): # used to flatten jacrev PyTree into one long gradient vector
    return torch.cat([t.reshape(-1) for t in d.values()])

# Simple solve for Ax=b
def solve_psd(A, b):
    return torch.linalg.solve(A, b)

def compute_krr_prediction(model, params, X_train, y_train, X_test, ridge):
# recall that for KRR, f(x)=f0(X) + grad(f(x))(theta-theta0)

    # run the model once on a single data point
    def f_single(p, x_single):
        out = functional_call(model, p, (x_single.unsqueeze(0),))  # [1,1]
        return out.squeeze()  # scalar

    with torch.no_grad():
        f0_train = functional_call(model, params, (X_train,))  # [N,1]
        f0_test  = functional_call(model, params, (X_test,))   # [M,1]

    grad_fn = jacrev(f_single)
    def per_ex_grad(x_single):
        g = grad_fn(params, x_single)
        return flatten_pytree(g)

    # jacobian matrix of the combined gradients
    J_train = vmap(per_ex_grad)(X_train)  # [N,P]
    J_test  = vmap(per_ex_grad)(X_test)   # [M,P]

    # NTK Gram matrix
    K_train = J_train @ J_train.T
    K_test_train = J_test @ J_train.T

    n = X_train.shape[0]
    Kreg = K_train + ridge * torch.eye(n, dtype=dtype, device=device)

    # solve KRR for alpha using residuals
    alpha = solve_psd(Kreg, y_train - f0_train)

    y_train_krr = f0_train + K_train @ alpha
    y_test_krr  = f0_test  + K_test_train @ alpha
    return y_train_krr, y_test_krr

def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))

print("=== Setup ===")
print(f"device={device}  dtype={dtype}")
print(f"N_TRAIN={N_TRAIN}  N_TEST={N_TEST}")
print(f"Widths to sweep: {WIDTH_LIST}")
print(f"BASE_LR={BASE_LR} (lr=BASE_LR/width), TARGET_TIME={TARGET_TIME} (steps ~ TARGET_TIME*width/BASE_LR)")
print(f"ridge(KRR)={RIDGE}")
print("Saving plots to current directory\n")

width_values, rmse_nn_krr_values = [], []
fits = {}

for width in WIDTH_LIST:
    torch.manual_seed(SEED + width)

    lr = BASE_LR / width
    n_steps = int(round(TARGET_TIME / lr))  # so lr * steps ≈ TARGET_TIME

    print(f"--- WIDTH {width} ---")
    print(f"lr={lr:.3e}, steps={n_steps} (effective_time≈{lr*n_steps:.3f})")

    # initialize model and use the same params from NTK
    base_model = MLP(width).to(device=device, dtype=dtype)
    params = {name: p.detach().clone().requires_grad_(True) for name, p in base_model.named_parameters()}

    print("Computing empirical NTK and KRR solution...")
    _, y_test_krr = compute_krr_prediction(base_model, params, X_train, y_train, X_test, RIDGE)
    print("  done.")

    # train NN starting from same init
    train_model = MLP(width).to(device=device, dtype=dtype)
    train_model.load_state_dict(base_model.state_dict())
    opt = torch.optim.SGD(train_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("Training NN...")
    for step in range(n_steps):
        opt.zero_grad()
        pred = train_model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        # print stats during training at set intervals
        if step in [0, n_steps//5, 2*n_steps//5, 3*n_steps//5, 4*n_steps//5, n_steps-1]:
            with torch.no_grad():
                y_nn_test = train_model(X_test)
                cur = rmse(y_nn_test[domain_mask], y_test_krr[domain_mask]).item()
            print(f"  step {step+1:6d}/{n_steps}: train_loss={loss.item():.3e}, RMSE(NN,KRR)_test={cur:.3e}")

    with torch.no_grad():
        y_nn_test = train_model(X_test)
        final_rmse = rmse(y_nn_test[domain_mask], y_test_krr[domain_mask]).item()

    print(f"Final RMSE(NN,KRR) test: {final_rmse:.3e}\n")
    width_values.append(width)
    rmse_nn_krr_values.append(final_rmse)

    fits[width] = {
        "X_test": X_test.detach().cpu().numpy(),
        "true": y_test_true.detach().cpu().numpy(),
        "krr": y_test_krr.detach().cpu().numpy(),
        "nn": y_nn_test.detach().cpu().numpy(),
    }

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.flatten()

for ax, width in zip(axes_flat[:-1], WIDTH_LIST):
    curves = fits[width]
    ax.plot(curves["X_test"], curves["true"], label="True", linewidth=2)
    ax.plot(curves["X_test"], curves["krr"], label="KRR (NTK@init)")
    ax.plot(curves["X_test"], curves["nn"], "--", label="NN GD")
    ax.scatter(X_train_np.squeeze(), y_train_np.squeeze(), s=20, alpha=0.5, label="Train pts")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Fit at width {width}")
    if width == WIDTH_LIST[0]:
        ax.legend()

rmse_ax = axes_flat[-1]
rmse_ax.plot(width_values, rmse_nn_krr_values, marker="o")
rmse_ax.set_xscale("log"); rmse_ax.set_yscale("log")
rmse_ax.set_xlabel("Width"); rmse_ax.set_ylabel("RMSE(NN, KRR) on test (in-domain)")
rmse_ax.set_title("RMSE(NN,KRR) vs width")

fig.tight_layout()
plt.savefig("fits_and_rmse.png")
plt.close(fig)

print("Done. Plot saved as fits_and_rmse.png")

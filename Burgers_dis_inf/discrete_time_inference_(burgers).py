!mkdir -p Data

# Download the file using wget and save it in the data directory
!wget https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/burgers_shock.mat -O Data/burgers_shock.mat

!mkdir PINNs
!cd PINNs
!git init
!git remote add origin https://github.com/maziarraissi/PINNs.git
!git config core.sparseCheckout true
!echo "Utilities/IRK_weights/" >> .git/info/sparse-checkout
!git pull origin master
!git sparse-checkout reapply

import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q):
        super(PhysicsInformedNN, self).__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)

        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
        self.u0 = torch.tensor(u0, dtype=torch.float32, requires_grad=True)

        self.dt = dt
        self.q = max(q, 1)

        self.layers = layers
        self.build_network(layers)

        # Load IRK weights
        IRK_weights = np.loadtxt(f'Utilities/IRK_weights/Butcher_IRK{q}.txt', ndmin=2)
        self.IRK_weights = torch.tensor(IRK_weights[:q**2+q].reshape((q+1, q)), dtype=torch.float32)
        self.IRK_times = torch.tensor(IRK_weights[q**2+q:], dtype=torch.float32)

        self.optimizer = optim.LBFGS(self.parameters(), lr=1, max_iter=50000, max_eval=50000, tolerance_grad=1.0 * np.finfo(float).eps)
        self.optimizer_adam = optim.Adam(self.parameters())

    def build_network(self, layers):
        modules = []
        num_layers = len(layers)
        for i in range(num_layers - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < num_layers - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        # Scaling input to [-1,1]
        x_scaled = 2 * (x - self.lb) / (self.ub - self.lb) - 1
        return self.net(x_scaled)

    def loss_func(self, pred, target):
        return torch.mean((pred - target)**2)

    def train(self, n_iter):
        for epoch in range(n_iter):
            def closure():
                self.optimizer_adam.zero_grad()
                U0_pred = self.forward(self.x0)
                loss = self.loss_func(U0_pred, self.u0)
                loss.backward()
                return loss

            self.optimizer_adam.step(closure)
            if epoch % 10000 == 0:
                loss_value = closure()
                print(f'Epoch {epoch}, Loss: {loss_value.item()}')

        def lbfgs_closure():
            self.optimizer.zero_grad()
            U0_pred = self.forward(self.x0)
            loss = self.loss_func(U0_pred, self.u0)
            loss.backward()
            return loss

        self.optimizer.step(lbfgs_closure)

    def predict(self, x_star):
        with torch.no_grad():
            x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True)
            return self.forward(x_star)

import numpy as np
import scipy.io
import torch

# Assuming the PyTorch model class PhysicsInformedNN is already imported

q = 500
layers = [1, 50, 50, 50, q + 1]
lb = np.array([-1.0])
ub = np.array([1.0])

N = 250

data = scipy.io.loadmat('Data/burgers_shock.mat')

t = data['t'].flatten()[:, None]  # T x 1
x = data['x'].flatten()[:, None]  # N x 1
Exact = np.real(data['usol']).T  # T x N

idx_t0 = 10
idx_t1 = 90
dt = t[idx_t1] - t[idx_t0]

# Initial data
noise_u0 = 0.0
idx_x = np.random.choice(Exact.shape[1], N, replace=False)
x0 = x[idx_x, :]
u0 = Exact[idx_t0:idx_t0+1, idx_x].T
u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

# Boundary data
x1 = np.vstack((lb, ub))

# Test data
x_star = x

# Convert data to PyTorch tensors
x0_torch = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
u0_torch = torch.tensor(u0, dtype=torch.float32, requires_grad=True)
x1_torch = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
x_star_torch = torch.tensor(x_star, dtype=torch.float32, requires_grad=True)

# Model initialization
model = PhysicsInformedNN(x0_torch, u0_torch, x1_torch, layers, dt, lb, ub, q)

# Training
model.train(50000)

# Prediction
U1_pred = model.predict(x_star_torch)

# Error calculation
error = torch.linalg.norm(U1_pred[:, -1] - torch.tensor(Exact[idx_t1, :], dtype=torch.float32), 2) / torch.linalg.norm(torch.tensor(Exact[idx_t1, :], dtype=torch.float32), 2)
print(f'Error: {error.item():e}')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Assuming all the required data (Exact, x, t, U1_pred) has been defined as per the previous context

# Create a new figure with an adjusted size
fig = plt.figure(figsize=(10, 6))

# Create a grid for the subplots
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# The main plot (top)
ax0 = plt.subplot(gs[0, :])
h = ax0.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
               extent=[t.min(), t.max(), x.min(), x.max()],
               origin='lower', aspect='auto')
# Create an axis on the right side for the colorbar
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

# Vertical lines to indicate the position of the slices
ax0.axvline(x=t[idx_t0], color='white', linestyle='-', linewidth=1)
ax0.axvline(x=t[idx_t1], color='white', linestyle='-', linewidth=1)
ax0.set_xlabel('$t$')
ax0.set_ylabel('$x$')
ax0.set_title('$u(t,x)$', fontsize=10)

# Lower left plot (slice at t[idx_t0])
ax1 = plt.subplot(gs[1, 0])

# Only plot the `x` values that correspond to the `u0` data points
# Assuming `x0` contains the sampled x locations that correspond to `u0`
ax1.plot(x0, u0, 'rx', markersize=6, label='Data')  # Corrected line here

# Now plotting the full 'Exact' profile for comparison
ax1.plot(x, Exact[idx_t0, :], 'b-', linewidth=2, label='Exact')

ax1.set_xlabel('$x$')
ax1.set_ylabel('$u(t,x)$')
ax1.set_title('$t = %.2f$' % (t[idx_t0]), fontsize=10)
ax1.set_xlim([x.min(), x.max()])
ax1.legend()

# Lower right plot (slice at t[idx_t1])
ax2 = plt.subplot(gs[1, 1])
ax2.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
ax2.plot(x_star, U1_pred[:,-1], 'r--', linewidth=2, label='Prediction')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$u(t,x)$')
ax2.set_title('$t = %.2f$' % (t[idx_t1]), fontsize=10)
ax2.set_xlim([x.min(), x.max()])
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Uncomment the next line if you want to save the figure
plt.savefig('./figures/Burgers.png')

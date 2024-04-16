!mkdir -p Data
!wget https://github.com/maziarraissi/PINNs/raw/master/main/Data/NLS.mat -O Data/NLS.mat
!pip install pyDOE

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub, device=device):
        super(PhysicsInformedNN, self).__init__()

        self.device = device

        # Normalize inputs
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        X0 = torch.cat((x0, torch.zeros_like(x0)), dim=1)
        X_lb = torch.cat((torch.zeros_like(tb) + lb[0], tb), dim=1)
        X_ub = torch.cat((torch.zeros_like(tb) + ub[0], tb), dim=1)

        self.x0 = X0[:, 0:1].to(device)
        self.t0 = X0[:, 1:2].to(device)

        self.x_lb = X_lb[:, 0:1].to(device)
        self.t_lb = X_lb[:, 1:2].to(device)

        self.x_ub = X_ub[:, 0:1].to(device)
        self.t_ub = X_ub[:, 1:2].to(device)

        self.x_f = X_f[:, 0:1].to(device)
        self.t_f = X_f[:, 1:2].to(device)

        self.u0 = u0.to(device)
        self.v0 = v0.to(device)

        # Define network
        self.layers = layers
        self.neural_net = self._build_network(layers).to(device)

        # Define optimizer
        self.optimizer = optim.LBFGS(self.parameters(),
                                     lr=1,
                                     max_iter=50000,
                                     max_eval=50000,
                                     tolerance_grad=1.0 * np.finfo(float).eps)
        self.optimizer_adam = optim.Adam(self.parameters())

    def _build_network(self, layers):
        """Utility method to build the neural network."""
        network = []
        num_layers = len(layers)
        for i in range(num_layers - 1):
            network.append(nn.Linear(layers[i], layers[i + 1]))
            if i < num_layers - 2:
                network.append(nn.Tanh())
        return nn.Sequential(*network)

    def forward(self, x, t):
        """Forward pass for both u and v predictions."""
        x = x.requires_grad_(True) if not x.requires_grad else x
        t = t.requires_grad_(True) if not t.requires_grad else t
        X = torch.cat([x, t], dim=1)

        uv = self.neural_net(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        """Calculates the f terms for the PDE."""
        u, v, u_x, v_x = self(x, t)

        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]

        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u

        return f_u, f_v

    def loss_function(self, u_pred, v_pred, f_u_pred, f_v_pred):
        """Calculate the loss function based on predictions and boundary conditions."""
        loss_u0 = torch.mean((self.u0 - u_pred)**2)
        loss_v0 = torch.mean((self.v0 - v_pred)**2)
        loss_f_u = torch.mean(f_u_pred**2)
        loss_f_v = torch.mean(f_v_pred**2)

        return loss_u0 + loss_v0 + loss_f_u + loss_f_v

    # first use ADAM for nIter (50000), then use LBFGS (quasi-Newton) for 50000 iter
    def train_model(self, nIter):
      self.train()

      # Use Adam for initial iterations
      for it in range(nIter):
          self.optimizer_adam.zero_grad()

          u0_pred, v0_pred, _, _ = self(self.x0, self.t0)
          f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)

          loss = self.loss_function(u0_pred, v0_pred, f_u_pred, f_v_pred)
          loss.backward()
          self.optimizer_adam.step()

          if it % 10 == 0:
              print(f'Iteration {it}: Loss = {loss.item()}')

      # Switch to LBFGS for fine-tuning
      def closure():
          self.optimizer.zero_grad()

          u0_pred, v0_pred, _, _ = self(self.x0, self.t0)
          f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)

          loss = self.loss_function(u0_pred, v0_pred, f_u_pred, f_v_pred)
          loss.backward()

          return loss

      self.optimizer.step(closure)

    def predict(self, X_star):
      self.eval()
      x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32).to(self.device)
      t_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32).to(self.device)
      u_star, v_star, _, _ = self(x_star, t_star)
      f_u_star, f_v_star = self.net_f_uv(x_star, t_star)

      return u_star.detach().cpu().numpy(), v_star.detach().cpu().numpy(), f_u_star.detach().cpu().numpy(), f_v_star.detach().cpu().numpy()

if __name__ == "__main__":
    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]

    # Load data
    data = scipy.io.loadmat('Data/NLS.mat')

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    h_star = Exact_h.T.flatten()[:,None]

    # Sampling initial and boundary data
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = torch.tensor(x[idx_x,:], dtype=torch.float32, device=device, requires_grad=True)
    u0 = torch.tensor(Exact_u[idx_x,0:1], dtype=torch.float32, device=device, requires_grad=True)
    v0 = torch.tensor(Exact_v[idx_x,0:1], dtype=torch.float32, device=device, requires_grad=True)

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = torch.tensor(t[idx_t,:], dtype=torch.float32, device=device, requires_grad=True)

    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_f = torch.tensor(X_f, dtype=torch.float32, device=device, requires_grad=True)

    # Initialize model
    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub, device)

    # Train model
    start_time = time.time()
    model.train_model(50000) #
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    # Prediction
    X_star_tensor = torch.tensor(X_star, dtype=torch.float32, device=device)
    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star_tensor)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    # Calculate errors
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error h: %e' % error_h)

    # Interpolation for plotting
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

    # Plot for |h(t, x)|
    ax0 = fig.add_subplot(gs[0, :])
    h = ax0.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax0.set_xlabel('$t$')
    ax0.set_ylabel('$x$')
    ax0.set_title('$|h(t,x)|$', fontsize=10)

    # Plot specific time slices
    times = [75, 100, 125]  # example times for slices
    for i, time in enumerate(times):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x.flatten(), Exact_h[:, time], 'b-', linewidth=2, label='Exact')
        ax.plot(x.flatten(), H_pred[time, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.set_title(f'$t = {t[time][0]:.2f}$', fontsize=10)
        ax.axis('square')
        ax.set_xlim([-5.1, 5.1])
        ax.set_ylim([-0.1, 5.1])
        if i == 2:  # Add legend only to the last plot to save space
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    plt.tight_layout()
    # plt.show()
    plt.savefig('NLS.png')


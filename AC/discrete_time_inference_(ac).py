!mkdir -p Data
!wget https://github.com/maziarraissi/PINNs/raw/master/main/Data/AC.mat -O Data/AC.mat

# download IRK_weights
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
import torch.nn.functional as F
import numpy as np
import scipy.io
# Setting the random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q):
        super(PhysicsInformedNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Scaling bounds
        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)

        # Training data
        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(self.device)
        self.u0 = torch.tensor(u0, dtype=torch.float32, requires_grad=True).to(self.device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(self.device)

        self.dt = dt
        self.q = max(q, 1)

        # Neural network layers
        self.layers = nn.ModuleList()
        num_layers = len(layers)
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # IRK weights
        IRK_weights = np.loadtxt('Utilities/IRK_weights/Butcher_IRK%d.txt' % q, ndmin=2)
        self.IRK_weights = torch.tensor(IRK_weights[:q**2+q], dtype=torch.float32).view(q+1, q).to(self.device)
        self.IRK_times = torch.tensor(IRK_weights[q**2+q:], dtype=torch.float32).to(self.device)

        # Optimizers
        self.optimizer = torch.optim.LBFGS(self.parameters(), lr=1, max_iter=50000, tolerance_grad=1e-7, tolerance_change=1e-9)
        self.adam_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        H = 2 * (x - self.lb) / (self.ub - self.lb) - 1
        for layer in self.layers[:-1]:
            H = torch.tanh(layer(H))
        Y = self.layers[-1](H)
        return Y

    def net_U0(self, x):
        U1 = self.forward(x)
        U = U1[:, :-1]
        U_x = torch.autograd.grad(U, x, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        U_xx = torch.autograd.grad(U_x, x, grad_outputs=torch.ones_like(U_x), create_graph=True)[0]
        F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_xx
        U0 = U1 - self.dt * torch.matmul(F, self.IRK_weights.T)
        return U0

    def net_U1(self, x):
        U1 = self.forward(x)
        U1_x = torch.autograd.grad(U1, x, grad_outputs=torch.ones_like(U1), create_graph=True)[0]
        return U1, U1_x

    def loss_func(self):
        U0_pred = self.net_U0(self.x0)
        U1_pred, U1_x_pred = self.net_U1(self.x1)
        loss_U0 = torch.mean((self.u0 - U0_pred)**2)
        loss_U1 = torch.mean((U1_pred[:-1] - U1_pred[1:])**2)
        loss_U1_x = torch.mean((U1_x_pred[:-1] - U1_x_pred[1:])**2)
        return loss_U0 + loss_U1 + loss_U1_x

    def train(self, nIter):
        for it in range(nIter):
            def closure():
                self.optimizer.zero_grad()
                loss = self.loss_func()
                loss.backward()
                return loss
            self.optimizer.step(closure)
            if it % 100 == 0:
                with torch.no_grad():
                    loss_value = self.loss_func()
                    print(f'Iter: {it}, Loss: {loss_value.item()}')

    def predict(self, x_star):
        x_star = torch.tensor(x_star, dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            U1_star = self.forward(x_star)
        return U1_star.cpu().numpy()

if __name__ == "__main__":

    data = scipy.io.loadmat('Data/AC.mat')
    t = data['tt'].flatten()[:, None]  # T x 1
    x = data['x'].flatten()[:, None]  # N x 1
    Exact = np.real(data['uu']).T  # T x N

    idx_t0 = 20
    idx_t1 = 180
    dt = t[idx_t1] - t[idx_t0]

    # Configuration parameters
    q = 100
    layers = [1, 200, 200, 200, 200, q + 1]
    lb = np.array([-1.0])
    ub = np.array([1.0])

    N = 200

    # Prepare initial data
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_t0:idx_t0 + 1, idx_x].T
    u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    # Boundary data
    x1 = np.vstack((lb, ub))

    # Convert all NumPy arrays to PyTorch tensors
    x0 = torch.from_numpy(x0).float().to(device)
    u0 = torch.from_numpy(u0).float().to(device)
    x1 = torch.from_numpy(x1).float().to(device)
    x_star = torch.from_numpy(x).float().to(device)

    # Initialize the model
    model = PhysicsInformedNN(x0, u0, x1, layers, dt.item(), lb, ub, q)
    model.to(device)

    # Training
    model.train(10000)

    # Prediction
    U1_pred = model.predict(x_star.cpu().numpy())  # Need to move data to CPU if model and data are on GPU

    # Error calculation
    error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    print('Error: %e' % (error))


    ############################# Plotting ###############################
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax0 = plt.subplot(gs0[:, :])

    h = ax0.imshow(Exact.T, interpolation='nearest', cmap='seismic',
                  extent=[t.min(), t.max(), x_star.min(), x_star.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax0.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax0.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth=1)

    ax0.set_xlabel('$t$')
    ax0.set_ylabel('$x$')
    ax0.set_title('$u(t,x)$', fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax1 = plt.subplot(gs1[0, 0])
    ax1.plot(x, Exact[idx_t0, :], 'b-', linewidth=2, label='Exact')
    ax1.plot(x0.numpy().flatten(), u0.numpy().flatten(), 'rx', linewidth=2, label='Data')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$u(t,x)$')
    ax1.set_title('$t = %.2f$' % (t[idx_t0]), fontsize=10)
    ax1.set_xlim([lb-0.1, ub+0.1])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

    ax2 = plt.subplot(gs1[0, 1])
    ax2.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
    ax2.plot(x_star.numpy().flatten(), U1_pred[:, -1], 'r--', linewidth=2, label='Prediction')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$u(t,x)$')
    ax2.set_title('$t = %.2f$' % (t[idx_t1]), fontsize=10)
    ax2.set_xlim([lb-0.1, ub+0.1])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

    # plt.show()
    plt.savefig("AC.pdf")


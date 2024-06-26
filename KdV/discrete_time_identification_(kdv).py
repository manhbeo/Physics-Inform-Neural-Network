!mkdir -p Data
!wget https://github.com/maziarraissi/PINNs/raw/master/main/Data/KdV.mat -O Data/KdV.mat

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, q, device=device):
        super(PhysicsInformedNN, self).__init__()
        self.device = device
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(device)

        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.u1 = torch.tensor(u1, dtype=torch.float32).to(device)

        self.dt = torch.tensor(dt, dtype=torch.float32).to(device)
        self.q = max(q, 1)

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = nn.Parameter(torch.zeros(1, device=device))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], device=device))

        # Load IRK weights
        tmp = np.loadtxt('Utilities/IRK_weights/Butcher_IRK%d.txt' % q, ndmin=2).astype(np.float32)
        weights = np.reshape(tmp[0:self.q**2+self.q], (self.q+1, self.q))
        self.IRK_alpha = torch.tensor(weights[:-1, :], dtype=torch.float32).to(device)
        self.IRK_beta = torch.tensor(weights[-1:, :], dtype=torch.float32).to(device)
        self.IRK_times = torch.tensor(tmp[self.q**2+self.q:], dtype=torch.float32).to(device)

    def initialize_NN(self, layers):
        weights = nn.ParameterList()
        biases = nn.ParameterList()
        num_layers = len(layers)
        for l in range(num_layers - 1):
            W = nn.Parameter(torch.randn(layers[l], layers[l+1]) * np.sqrt(2 / (layers[l] + layers[l+1])))
            b = nn.Parameter(torch.zeros(1, layers[l+1]))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def forward(self, x):
        # Normalize input
        x = 2 * (x - self.lb) / (self.ub - self.lb) - 1
        for i in range(len(self.weights) - 1):
            # print(x.shape)
            # print(self.weights[i].shape)
            # print(self.biases[i].shape)
            x = torch.tanh(torch.addmm(self.biases[i], x, self.weights[i]))
        x = F.linear(x, self.weights[-1], self.biases[-1])
        return x

    def net_U0(self, x):
        U = self.forward(x)
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        U_xxx = self.fwd_gradients(U_xx, x)
        F = -self.lambda_1 * U * U_x - torch.exp(self.lambda_2) * U_xxx
        U0 = U - self.dt * torch.matmul(F, self.IRK_alpha.t())
        return U0

    def net_U1(self, x):
        U = self.forward(x)
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        U_xxx = self.fwd_gradients(U_xx, x)
        F = -self.lambda_1 * U * U_x - torch.exp(self.lambda_2) * U_xxx
        U1 = U + self.dt * torch.matmul(F, (self.IRK_beta - self.IRK_alpha).t())
        return U1

    def fwd_gradients(self, y, x):
        dummy = torch.ones_like(y, requires_grad=True).to(self.device)
        grad_yx = torch.autograd.grad(y, x, grad_outputs=dummy, create_graph=True)[0]
        return grad_yx

    def loss_func(self, u_pred0, u_pred1):
        loss0 = torch.mean((self.u0 - u_pred0) ** 2)
        loss1 = torch.mean((self.u1 - u_pred1) ** 2)
        return loss0 + loss1

    def train(self, nIter):
        # Adam optimizer for initial iterations
        optimizer_Adam = torch.optim.Adam(self.parameters(), lr=1e-3)
        for it in range(nIter):
            optimizer_Adam.zero_grad()
            U0_pred = self.net_U0(self.x0)
            U1_pred = self.net_U1(self.x1)
            loss = self.loss_func(U0_pred, U1_pred)
            loss.backward()
            optimizer_Adam.step()

            if it % 100 == 0:
                print(f'Iteration {it}, Loss: {loss.item()}')

        # Switch to L-BFGS optimizer for finer convergence
        optimizer_LBFGS = torch.optim.LBFGS(self.parameters(), lr=1, max_iter=50000, tolerance_grad=1.0*np.finfo(float).eps)

        def closure():
            optimizer_LBFGS.zero_grad()
            U0_pred = self.net_U0(self.x0)
            U1_pred = self.net_U1(self.x1)
            loss = self.loss_func(U0_pred, U1_pred)
            loss.backward()
            return loss

        optimizer_LBFGS.step(closure)

    def predict(self, x_star):
        x_star_tensor = torch.tensor(x_star, dtype=torch.float32).to(self.device)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            U0_star = self.net_U0(x_star_tensor)
            U1_star = self.net_U1(x_star_tensor)
        return U0_star.cpu().numpy(), U1_star.cpu().numpy()

if __name__ == "__main__":
    q = 50
    skip = 120

    N0 = 199
    N1 = 201
    layers = [1, 50, 50, 50, 50, q]

    # Load data
    data = scipy.io.loadmat('Data/KdV.mat')

    t_star = torch.tensor(data['tt'].flatten()[:, None], dtype=torch.float32)
    x_star = torch.tensor(data['x'].flatten()[:, None], dtype=torch.float32)
    Exact = torch.tensor(np.real(data['uu']), dtype=torch.float32)

    idx_t = 40

    # Noiseless data setup
    noise = 0.0

    idx_x = torch.randperm(Exact.shape[0])[:N0]
    x0 = x_star[idx_x, :]
    u0 = Exact[idx_x, idx_t][:, None]

    idx_x = torch.randperm(Exact.shape[0])[:N1]
    x1 = x_star[idx_x, :]
    u1 = Exact[idx_x, idx_t + skip][:, None]

    dt = (t_star[idx_t + skip] - t_star[idx_t]).item()

    # Domain bounds
    lb = x_star.min(0).values.item()
    ub = x_star.max(0).values.item()

    model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)
    model.train(nIter=50000)

    U0_pred, U1_pred = model.predict(x_star)

    lambda_1_value = model.lambda_1.item()
    lambda_2_value = torch.exp(model.lambda_2).item()

    error_lambda_1 = np.abs(lambda_1_value - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.0025) / 0.0025 * 100

    print('Error lambda_1: {:.2f}%'.format(error_lambda_1))
    print('Error lambda_2: {:.2f}%'.format(error_lambda_2))

    # Noisy data setup
    noise = 0.01

    u0 += noise * u0.std() * torch.randn(u0.shape)
    u1 += noise * u1.std() * torch.randn(u1.shape)

    model_noisy = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q, device='cuda')
    model_noisy.train(nIter=50000)

    U0_pred_noisy, U1_pred_noisy = model_noisy.predict(x_star)

    lambda_1_value_noisy = model_noisy.lambda_1.item()
    lambda_2_value_noisy = torch.exp(model_noisy.lambda_2).item()

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) / 1.0 * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.0025) / 0.0025 * 100

    print('Error lambda_1 (noisy): {:.2f}%'.format(error_lambda_1_noisy))
    print('Error lambda_2 (noisy): {:.2f}%'.format(error_lambda_2_noisy))


    ############################# Plotting ###############################
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=0.93, bottom=0.63, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(Exact.cpu().numpy(), interpolation='nearest', cmap='rainbow',
                  extent=[t_star.min(), t_star.max(), lb, ub],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(x_star.min(), x_star.max(), 2)[:, None]
    ax.plot(t_star[idx_t].item()*np.ones((2,1)), line, 'w-', linewidth=1.0)
    ax.plot(t_star[idx_t + skip].item()*np.ones((2,1)), line, 'w-', linewidth=1.0)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize=10)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.53, bottom=0.23, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_star.cpu().numpy(), Exact[:, idx_t].cpu().numpy(), 'b', linewidth=2, label='Exact')
    ax.plot(x0.cpu().numpy(), u0.cpu().numpy(), 'rx', linewidth=2, label='Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title(f'$t = {t_star[idx_t].item():.2f}$\n{u0.shape[0]} training data', fontsize=10)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_star.cpu().numpy(), Exact[:, idx_t + skip].cpu().numpy(), 'b', linewidth=2, label='Exact')
    ax.plot(x1.cpu().numpy(), u1.cpu().numpy(), 'rx', linewidth=2, label='Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title(f'$t = {t_star[idx_t + skip].item():.2f}$\n{u1.shape[0]} training data', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=0.13, bottom=0.05, left=0.15, right=0.85, wspace=0.0)

    ax = plt.subplot(gs2[0, 0])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + 0.0025 u_{xxx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1 + s2 + s3 + s4 + s5
    ax.text(0, 0.5, s, fontsize=10)

    plt.show()
    plt.savefig('Kdv.pdf')

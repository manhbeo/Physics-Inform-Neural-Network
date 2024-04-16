!mkdir -p Data

# Download the file using wget and save it in the data directory
!wget https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_t0_vorticity.mat -O Data/cylinder_nektar_t0_vorticity.mat
!wget https://github.com/maziarraissi/PINNs/raw/master/main/Data/cylinder_nektar_wake.mat -O Data/cylinder_nektar_wake.mat

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PhysicsInformedNN(nn.Module):
    def __init__(self, x, y, t, u, v, layers):
      super(PhysicsInformedNN, self).__init__()
      self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
      self.y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
      self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
      self.u = torch.tensor(u, dtype=torch.float32).to(device)
      self.v = torch.tensor(v, dtype=torch.float32).to(device)

      self.layers = nn.ModuleList()
      num_layers = len(layers)
      for i in range(num_layers - 1):
          self.layers.append(nn.Linear(layers[i], layers[i + 1]).to(device))

      self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)).to(device)
      self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)).to(device)

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            X = torch.tanh(layer(X))
        output = self.layers[-1](X)
        psi = output[:, [0]]
        p = output[:, [1]]
        psi.requires_grad_(True)
        p.requires_grad_(True)
        return psi, p

    def net_NS(self, x, y, t):
        psi, p = self.forward(x, y, t)

        if not psi.requires_grad:
            raise RuntimeError("psi does not have gradients")
        if not p.requires_grad:
            raise RuntimeError("p does not have gradients")
        # Compute gradients for u and v
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        # Compute required gradients for Navier-Stokes terms
        ones_u = torch.ones(u.size(), device=u.device)
        ones_v = torch.ones(v.size(), device=v.device)
        u_t = torch.autograd.grad(u, t, grad_outputs=ones_u, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=ones_u, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=ones_u, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_u, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_u, create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, grad_outputs=ones_v, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=ones_v, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=ones_v, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_v, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_v, create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # Formulate the Navier-Stokes equations
        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v


    def loss_func(self, u_pred, v_pred, f_u_pred, f_v_pred):
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_v = torch.mean((self.v - v_pred) ** 2)
        loss_f_u = torch.mean(f_u_pred ** 2)
        loss_f_v = torch.mean(f_v_pred ** 2)
        return loss_u + loss_v + loss_f_u + loss_f_v

    def train(self, nIter=1000, nIter_LBFGS=500):
      # Adam optimization
      optimizer_adam = optim.Adam(self.parameters(), lr=0.001)
      for it in range(nIter):
          optimizer_adam.zero_grad()
          u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
          loss = self.loss_func(u_pred, v_pred, f_u_pred, f_v_pred)
          loss.backward()
          optimizer_adam.step()
          if it % 1000 == 0:
              print(f'Iter {it}, Loss {loss.item()}')
      # Switch to L-BFGS optimizer
      optimizer_lbfgs = optim.LBFGS(self.parameters(), max_iter=20, line_search_fn='strong_wolfe')
      def closure():
          optimizer_lbfgs.zero_grad()
          u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
          loss = self.loss_func(u_pred, v_pred, f_u_pred, f_v_pred)
          loss.backward()
          return loss
      for it in range(nIter_LBFGS):
          optimizer_lbfgs.step(closure)
          with torch.no_grad():
              loss = closure()
          if it % 1000 == 0 or it == nIter_LBFGS - 1:
              print(f'LBFGS Iter {it}, Loss {loss.item()}')


    def predict(self, x_star, y_star, t_star):
        self.eval()
        with torch.no_grad():
            u_star, v_star, p_star = self.net_NS(torch.tensor(x_star, dtype=torch.float32).to(device),
                                                 torch.tensor(y_star, dtype=torch.float32).to(device),
                                                 torch.tensor(t_star, dtype=torch.float32).to(device))
        return u_star.cpu().numpy(), v_star.cpu().numpy(), p_star.cpu().numpy()

def plot_solution(X_star, u_star, index):
    # Ensure u_star is a numpy array (useful if it's a PyTorch tensor)
    if not isinstance(u_star, np.ndarray):
        u_star = u_star.cpu().numpy()

    # Find the bounds of the plotting area
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    # Interpolate the u_star values onto the grid defined by X, Y
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    # Plotting
    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.title('Solution at Index {}'.format(index))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

if __name__ == "__main__":
  data = loadmat('Data/cylinder_nektar_wake.mat')
  U_star = data['U_star']  # N x 2 x T
  P_star = data['p_star']  # N x T
  t_star = data['t']       # T x 1
  X_star = data['X_star']  # N x 2

  N = X_star.shape[0]
  T = t_star.shape[0]

  # Rearrange Data
  XX = torch.tile(torch.tensor(X_star[:, 0:1]), (1, T)).to(device)
  YY = torch.tile(torch.tensor(X_star[:, 1:2]), (1, T)).to(device)
  TT = torch.tile(torch.tensor(t_star), (1, N)).T.to(device)

  UU = torch.tensor(U_star[:, 0, :]).to(device)
  VV = torch.tensor(U_star[:, 1, :]).to(device)
  PP = torch.tensor(P_star).to(device)

  x = XX.flatten()[:, None]
  y = YY.flatten()[:, None]
  t = TT.flatten()[:, None]

  u = UU.flatten()[:, None]
  v = VV.flatten()[:, None]
  p = PP.flatten()[:, None]

  ######################## Noiseless Data ###############################

  idx = torch.randperm(N * T)[:5000].to(device)
  x_train = x[idx]
  y_train = y[idx]
  t_train = t[idx]
  u_train = u[idx]
  v_train = v[idx]

  # Network training
  model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers).to(device)
  model.train(1) #20000

  # Test data preparation
  snap = torch.tensor([100])
  x_star = torch.tensor(X_star[:, 0:1]).to(device)
  y_star = torch.tensor(X_star[:, 1:2]).to(device)
  t_star = TT[:, snap].to(device)

  u_star = torch.tensor(U_star[:, 0, snap]).to(device)
  v_star = torch.tensor(U_star[:, 1, snap]).to(device)
  p_star = torch.tensor(P_star[:, snap]).to(device)

  # Prediction
  u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

  # Error calculation
  error_u = torch.linalg.norm(u_star - u_pred, 2) / torch.linalg.norm(u_star, 2)
  error_v = torch.linalg.norm(v_star - v_pred, 2) / torch.linalg.norm(v_star, 2)
  error_p = torch.linalg.norm(p_star - p_pred, 2) / torch.linalg.norm(p_star, 2)

  print(f'Error u: {error_u:e}')
  print(f'Error v: {error_v:e}')
  print(f'Error p: {error_p:e}')

  ########################### Noisy Data ###############################
  noise = 0.01
  u_noise = noise * torch.std(u_train) * torch.randn(u_train.shape).to(device)
  v_noise = noise * torch.std(v_train) * torch.randn(v_train.shape).to(device)

  u_train_noisy = u_train + u_noise
  v_train_noisy = v_train + v_noise

  # Make sure to convert noisy data to tensors and move to the correct device, if not already
  u_train_noisy = u_train_noisy.to(device)
  v_train_noisy = v_train_noisy.to(device)

  # Training the model with noisy data
  model_noisy = PhysicsInformedNN(x_train, y_train, t_train, u_train_noisy, v_train_noisy, layers).to(device)
  model_noisy.train(1) #200000

  # Retrieving lambda values directly from the model parameters
  lambda_1_value_noisy = model_noisy.lambda_1.item()
  lambda_2_value_noisy = model_noisy.lambda_2.item()

  # Calculating errors
  error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
  error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01) / 0.01 * 100

  # Printing errors
  print(f'Error lambda_1 (noisy): {error_lambda_1_noisy:.5f}%')
  print(f'Error lambda_2 (noisy): {error_lambda_2_noisy:.5f}%')


  ############################# Plotting ###############################
  data_vort = scipy.io.loadmat('Data/cylinder_nektar_t0_vorticity.mat')
  x_vort = data_vort['x']
  y_vort = data_vort['y']
  w_vort = data_vort['w']
  modes = int(data_vort['modes'].item())
  nel = int(data_vort['nel'].item())

  # Reshape data
  xx_vort = np.reshape(x_vort, (modes + 1, modes + 1, nel), order='F')
  yy_vort = np.reshape(y_vort, (modes + 1, modes + 1, nel), order='F')
  ww_vort = np.reshape(w_vort, (modes + 1, modes + 1, nel), order='F')

  box_lb = np.array([1.0, -2.0])
  box_ub = np.array([8.0, 2.0])

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.axis('off')

  ####### Row 0: Vorticity ##################
  gs0 = gridspec.GridSpec(1, 2)
  gs0.update(top=0.95, bottom=0.55, left=0.0, right=1.0, wspace=0)
  ax = plt.subplot(gs0[:, :])

  for i in range(nel):
      h = ax.pcolormesh(xx_vort[:, :, i], yy_vort[:, :, i], ww_vort[:, :, i], cmap='seismic', shading='gouraud', vmin=-3, vmax=3)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(h, cax=cax)

  # Draw bounding box
  ax.plot([box_lb[0], box_lb[0]], [box_lb[1], box_ub[1]], 'k-', linewidth=1)
  ax.plot([box_ub[0], box_ub[0]], [box_lb[1], box_ub[1]], 'k-', linewidth=1)
  ax.plot([box_lb[0], box_ub[0]], [box_lb[1], box_lb[1]], 'k-', linewidth=1)
  ax.plot([box_lb[0], box_ub[0]], [box_ub[1], box_ub[1]], 'k-', linewidth=1)

  ax.set_aspect('equal', 'box')
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.set_title('Vorticity', fontsize=10)
  plt.show()

  fig = plt.figure(figsize=(15, 7))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_proj_type('ortho')  # Orthographic projection

  # Limits and ranges
  r1 = [x_train.min(), x_train.max()]
  r2 = [t_train.min(), t_train.max()]
  r3 = [y_train.min(), y_train.max()]

  # GridSpec layout
  gs1 = gridspec.GridSpec(1, 2)
  gs1.update(top=0.5, bottom=0.05, left=0.05, right=0.95, wspace=0.2)

  ####### Row 1: Training data and contour plots ##################

  # Subplot for u(t,x,y)
  ax1 = plt.subplot(gs1[:, 0], projection='3d')
  ax1.set_title('$u(t,x,y)$')
  ax1.scatter(x_train.cpu().detach().numpy(), t_train.cpu().detach().numpy(), y_train.cpu().detach().numpy(), s=0.1, color='r')
  # Creating a contour plot at the mean value of t
  U_contours = ax1.contourf(X, UU_star, Y, zdir='y', offset=t_train.mean(), cmap='viridis', alpha=0.8)
  ax1.set_xlabel('$x$')
  ax1.set_ylabel('$t$')
  ax1.set_zlabel('$y$')
  ax1.set_xlim(r1)
  ax1.set_ylim(r2)
  ax1.set_zlim(r3)

  # Subplot for v(t,x,y)
  ax2 = plt.subplot(gs1[:, 1], projection='3d')
  ax2.set_title('$v(t,x,y)$')
  ax2.scatter(x_train, t_train, y_train, s=0.1, color='b')
  # Creating a contour plot at the mean value of t
  V_contours = ax2.contourf(X, VV_star, Y, zdir='y', offset=t_train.mean(), cmap='viridis', alpha=0.8)
  ax2.set_xlabel('$x$')
  ax2.set_ylabel('$t$')
  ax2.set_zlabel('$y$')
  ax2.set_xlim(r1)
  ax2.set_ylim(r2)
  ax2.set_zlim(r3)

  # Make the axes of both plots equal
  axisEqual3D(ax1)
  axisEqual3D(ax2)

  plt.show()

  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_proj_type('ortho')  # Orthographic projection for a 3D plot

  # GridSpec for layout
  gs1 = gridspec.GridSpec(1, 1)
  gs1.update(top=1.0, bottom=0.1, left=0.05, right=0.95, wspace=0.1)

  ####### Plotting v(t, x, y) in 3D ##################
  ax = plt.subplot(gs1[:, :], projection='3d')
  ax.set_title('$v(t, x, y)$')

  # Scatter plot for training data
  ax.scatter(x_train.cpu().detach().numpy(), t_train.cpu().detach().numpy(), y_train.cpu().detach().numpy(), s=0.1, color='b', label='Training Data')

  # Contour plot embedded into the 3D plot at a fixed t-value
  # Adjust the offset to match the context of your data, here it's taken as the mean of t_train
  v_contours = ax.contourf(X, VV_star, Y, zdir='y', offset=t_train.mean(), cmap='viridis', alpha=0.8)

  # Setting labels and limits based on data
  ax.set_xlabel('$x$')
  ax.set_ylabel('$t$')
  ax.set_zlabel('$y$')

  # Setting the limits for x, t, and y dimensions
  r1 = [x_train.min(), x_train.max()]
  r2 = [t_train.min(), t_train.max()]
  r3 = [y_train.min(), y_train.max()]
  ax.set_xlim(r1)
  ax.set_ylim(r2)
  ax.set_zlim(r3)

  axisEqual3D(ax)

  # Adding a color bar for the contour plot
  colorbar = plt.colorbar(v_contours, ax=ax, shrink=0.5, aspect=5)
  colorbar.set_label('Velocity v')

  plt.legend()
  plt.show()
  
  plt.savefig("NavierStokes_data.pdf")


  fig = plt.figure(figsize=(12, 6))
  gs2 = gridspec.GridSpec(1, 2)
  gs2.update(top=1.0, bottom=0.1, left=0.1, right=0.9, wspace=0.5)
  

  ####### Row 2: Pressure Comparison ##################

  # Plot Predicted Pressure
  ax1 = plt.subplot(gs2[0, 0])
  im1 = ax1.imshow(PP_star, interpolation='nearest', cmap='rainbow',
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')
  divider1 = make_axes_locatable(ax1)
  cax1 = divider1.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(im1, cax=cax1, orientation='vertical')
  ax1.set_xlabel('$x$')
  ax1.set_ylabel('$y$')
  ax1.set_title('Predicted Pressure Field')

  # Plot Exact Pressure
  ax2 = plt.subplot(gs2[0, 1])
  im2 = ax2.imshow(P_exact, interpolation='nearest', cmap='rainbow',
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')
  divider2 = make_axes_locatable(ax2)
  cax2 = divider2.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(im2, cax=cax2, orientation='vertical')
  ax2.set_xlabel('$x$')
  ax2.set_ylabel('$y$')
  ax2.set_title('Exact Pressure Field')

  plt.show()
  
  plt.savefig("NavierStokes_prediction.pdf")

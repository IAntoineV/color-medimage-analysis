
import torch
from img_tools import total_variation_smoothing




def GD(H0, V, W, nb_iter, lr):

    nb_stain, h, w = H0.shape
    c, _ = W.shape

    V = V.reshape(c, -1)
    H = H0.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([H], lr=lr)
    for i in range(nb_iter):
        opt.zero_grad()
        H_reshaped = H.view(nb_stain, -1)
        mse = 1 / 2 * (V - W @ H_reshaped).norm() ** 2
        loss = mse
        loss.backward()
        opt.step()
    return H.detach()

def PGD(H0, V, W, nb_iter, lr, lambda1, lambda2, eps):

    nb_stain, h, w = H0.shape
    c, _ = W.shape

    V = V.reshape(c,-1)
    H = H0.clone().detach().requires_grad_(True)
    for i in range(nb_iter):
        H_reshaped = H.view(nb_stain, -1)
        mse = 1/2 * (V- W @ H_reshaped).norm()**2
        reg = lambda1/2 * H.norm()**2 + lambda2 * total_variation_smoothing(H.unsqueeze(0), EPS=eps)
        loss = mse + reg
        loss.backward()
        H = torch.relu(H - lr * H.grad).detach().requires_grad_(True)

    return H



def unrolled_PGD(H0, V, W, nb_iter, theta):

    nb_stain, h, w = H0.shape
    c, _ = W.shape

    V = V.reshape(c, -1)
    H = H0.clone().detach().requires_grad_(True)
    for i in range(nb_iter):
        lr, lambda1, lambda2, eps = theta[i]
        H_reshaped = H.view(nb_stain, -1)
        mse = 1 / 2 * (V - W @ H_reshaped).norm() ** 2
        reg = lambda1 / 2 * H.norm() ** 2 + lambda2 * total_variation_smoothing(H.unsqueeze(0), EPS=eps)
        loss = mse + reg
        loss.backward()
        H = torch.relu(H - lr * H.grad).detach().requires_grad_(True)

    return H



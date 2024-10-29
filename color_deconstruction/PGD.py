
import torch
from img_tools import total_variation_smoothing







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

    H = H0.clone().detach().requires_grad_(True)
    for i in range(nb_iter):
        lr,lambda1, lambda2, eps = theta[i]
        mse = 1 / 2 * (V - W @ H).norm() ** 2
        reg = lambda1 / 2 * H.norm() ** 2 + lambda2 * total_variation_smoothing(H, EPS=eps)
        loss = mse + reg
        loss.backward(inputs = H)
        H = torch.relu(H - lr * H.grad).detach().requires_grad_(True)
    return H

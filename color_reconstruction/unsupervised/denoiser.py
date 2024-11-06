import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Diffusion:
    def __init__(self, denoising_model, device=None,timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        if device is None:
            device = torch.device('cpu')
        self.device = device
        # Define a linear schedule for the betas
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # cumulative product of alphas
        self.model=denoising_model
    def forward_diffusion(self, x0, t):
        """
        Diffuse input image x0 to timestep t by adding gaussian noise.
        :param x0 (torch.Tensor): input image x0
        :param t (int): number of diffusion timesteps
        :return: x_t (torch.Tensor): Noisy image at timestep t.
        :return: noise (torch.Tensor): The noise added.
        """
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1) # Shape it for broadcasting
        noise = torch.randn_like(x0).to(self.device)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def backward_diffusion(self, xt, t, sampling=False):
        """

        :param xt (torch.Tensor): t times noised image
        :param t (torch.Tensor): number of diffusion timesteps
        :return:
        """
        t = t-1
        if isinstance(t, int):
            t= torch.tensor([t], dtype=torch.int, device=self.device)
        t = t.view(-1,1,1,1)
        epsilon_theta = self.model(xt)
        rescaling_coef_eps_theta = (self.betas[t]/torch.sqrt(1-self.alpha_bars[t])).view(-1, 1, 1, 1)
        last_rescale = (1/torch.sqrt(self.alphas[t])).view(-1,1,1,1)
        # Eq (11)
        x_denoised = last_rescale*(xt - epsilon_theta * rescaling_coef_eps_theta)
        if sampling:
            z = torch.randn_like(x_denoised).to(self.device)
            sigma_t = torch.sqrt(self.betas[t]).view(-1, 1, 1, 1)
            x_denoised = x_denoised +  (t>0)* sigma_t * z
        return x_denoised, epsilon_theta

    def sampling(self, shape, T=None):
        """
        Generate a new image

        :param t:
        :param shape:
        :return:
        """
        if T is None:
            T = self.timesteps
        xT = torch.randn(shape).to(self.device)
        xt = xT
        for denoising_step in range(T):
            timestep = T - denoising_step -1
            xt,_ = self.backward_diffusion(xt, timestep, sampling=True)
        x0 = xt
        return x0

    def train(self, dataloader, lr=1e-4, num_epochs=10, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        mse_loss = nn.MSELoss()
        device = self.device
        for epoch in range(num_epochs):
            for images, _ in dataloader:
                images = images.to(device)

                # Sample random timestep for each batch element
                t = torch.randint(0, self.timesteps, (images.size(0),), device=device)

                # Apply forward diffusion process to get noisy image at timestep `t`
                x_t, noise = self.forward_diffusion(images, t)

                # Predict noise from noisy image at timestep `t`
                _, noise_pred = self.backward_diffusion(x_t, t)

                # Compute loss
                loss = mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4g}")


class UNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        super(UNet, self).__init__()

        # Encoder layers
        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, base_channels, 3, padding=1), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU())

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, 3, padding=1), nn.ReLU())

        # Output layer
        self.output = nn.Conv2d(base_channels, input_channels, 1)

    def forward(self, x):
        # Encode
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool2d(x1, 2))

        # Bottleneck
        x_bottleneck = self.bottleneck(F.max_pool2d(x2, 2))

        # Decode
        x = F.interpolate(self.decoder1(x_bottleneck), scale_factor=2)
        x = F.interpolate(self.decoder2(x + x2), scale_factor=2)

        # Output layer
        return self.output(x + x1)



def main():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    # Set up data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize diffusion process and model
    timesteps = 1000
    device = torch.device("cuda")
    model = UNet().to(device)
    diffusion = Diffusion(model, timesteps=timesteps, device=device)


    # Train model
    diffusion.train(train_loader, num_epochs=50)
    shape = (1,3,32,32)
    img_gen = diffusion.sampling(shape)
    plt.imshow(np.clip(img_gen[0].permute(2,1,0).detach().cpu().numpy(), 0, 1))
    plt.show()


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn

class Diffusion:
    def __init__(self, denoising_model, W_stain, device=None,timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps
        if device is None:
            device = torch.device('cpu')
        self.device = device
        # Define a linear schedule for the betas
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # cumulative product of alphas
        self.model=denoising_model
        self.W_stain = W_stain.view(1,1,1,W_stain.shape[-1]).to(self.device)
    def forward_diffusion(self, x0, t):
        """
        Diffuse input image x0 to timestep t by adding gaussian noise.
        :param x0 (torch.Tensor): input image x0
        :param t (int): number of diffusion timesteps
        :return: x_t (torch.Tensor): Noisy image at timestep t.
        :return: noise (torch.Tensor): The noise added.
        """
        if x0.size()<=3:
            assert "Unbatched image in the diffusion process"
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1) # Shape it for broadcasting
        noise = torch.randn(x0.shape[:-1]).to(self.device).unsqueeze(-1) # shape (batch,w,l,1)
        x0 = x0.to(self.device)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise * self.W_stain
        #xt = torch.clip(xt, 0., 1.)
        return xt, noise

    def backward_diffusion(self, xt, t, sampling=False):
        """

        :param xt (torch.Tensor): t times noised image
        :param t (torch.Tensor): number of diffusion timesteps
        :return:
        """
        if len(xt.shape)==3:
            xt=xt.unsqueeze(0)
        if isinstance(t, int):
            t= torch.tensor([t], dtype=torch.int, device=self.device)
        t = t.view(-1,1,1,1)
        epsilon_theta = self.model(xt, t).unsqueeze(-1) # (batch_size,w,l, 1)
        rescaling_coef_eps_theta = (self.betas[t]/torch.sqrt(1-self.alpha_bars[t])).view(-1, 1, 1, 1)
        last_rescale = (1/torch.sqrt(self.alphas[t])).view(-1,1,1,1)
        # Eq (11)
        x_denoised = last_rescale*(xt - epsilon_theta  * self.W_stain * rescaling_coef_eps_theta)
        if sampling and t==0:
            return x_denoised, epsilon_theta
        if sampling:
            z = torch.randn(x_denoised.shape[:-1]).to(self.device).unsqueeze(-1)
            sigma_t = torch.sqrt(self.betas[t]).view(-1, 1, 1, 1)
            x_denoised = x_denoised +  sigma_t * z * self.W_stain
        return x_denoised, epsilon_theta


    def train(self, dataloader, lr=1e-4, num_epochs=10, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        mse_loss = nn.MSELoss()
        device = self.device
        for epoch in range(num_epochs):
            i=0
            for images in dataloader:
                i+=1
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
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4g}, nb_step: {i}")



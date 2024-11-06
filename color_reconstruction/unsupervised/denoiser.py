import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Diffusion:
    def __init__(self, denoising_model, device=None,timesteps=1000, beta_start=1e-4, beta_end=2e-2):
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
        x0 = x0.to(self.device)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        xt = torch.clip(xt, 0., 1.)
        return xt, noise

    def backward_diffusion(self, xt, t, sampling=False):
        """

        :param xt (torch.Tensor): t times noised image
        :param t (torch.Tensor): number of diffusion timesteps
        :return:
        """
        if len(xt.shape)==3:
            xt=xt.unsqueeze(0)
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

    def sampling(self, shape=None, xT=None,T=None):
        """
        Generate a new image

        :param t:
        :param shape:
        :return:
        """
        assert shape is not None or xT is not None, "No specific prior or noise shape given !"

        if T is None:
            T = self.timesteps
        if xT is None:
            xT = torch.randn(shape)
        xt = xT.to(self.device)
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
            for images in dataloader:
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



def main_2():
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader

    # Chargement du dataset avec extraction de patchs
    class HistoLiverPatchDataset(Dataset):
        def __init__(self, image_dir, patch_size=64, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.patch_size = patch_size
            self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if
                           img.endswith('.jpg') or img.endswith('.png')]

        def __len__(self):
            return len(self.images) * (64 // self.patch_size) ** 2  # Nombre de patchs total

        def __getitem__(self, idx):
            img_idx = idx // ((64 // self.patch_size) ** 2)  # Sélection de l'image
            patch_idx = idx % ((64 // self.patch_size) ** 2)  # Sélection du patch

            img_path = self.images[img_idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            # Extraire le patch
            row = (patch_idx // (64 // self.patch_size)) * self.patch_size
            col = (patch_idx % (64 // self.patch_size)) * self.patch_size
            patch = image[:, row:row + self.patch_size, col:col + self.patch_size]

            return patch

    # Définir les transformations pour les images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Redimensionner les images en 64x64
        transforms.ToTensor()
    ])

    # Charger le dataset de patchs
    image_dir = '../../dataset/data/liver_HES'  # Assurez-vous que ce dossier contient vos images
    patch_size = 64
    dataset = HistoLiverPatchDataset(image_dir=image_dir, patch_size=patch_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Batch de 16 patchs

    timesteps = 20
    device = torch.device("cuda")
    model = UNet().to(device)
    diffusion = Diffusion(model, timesteps=timesteps, beta_end=1e-3,device=device)
    diffusion.train(dataloader, lr=1e-3, num_epochs=500)

    # Chargement d'une image complète et affichage des résultats
    full_image = transform(Image.open(dataset.images[0]).convert("RGB"))  # Charger une image

    noisy_img, _ = diffusion.forward_diffusion(full_image, 10)
    noisy_img = noisy_img.squeeze(0)
    reconstructed_full_image = diffusion.sampling(shape=None,xT=noisy_img, T=10).squeeze(0).clip(0.,1.)

    # Afficher les résultats
    def show_images(original, noisy, reconstructed):
        original = np.transpose(original.cpu().numpy(), (1, 2, 0))
        noisy = np.transpose(noisy.cpu().numpy(), (1, 2, 0))
        reconstructed = np.transpose(reconstructed.detach().cpu().numpy(), (1, 2, 0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title("Image Originale")
        axes[0].axis("off")

        axes[1].imshow(noisy)
        axes[1].set_title("Image Bruitée")
        axes[1].axis("off")

        axes[2].imshow(reconstructed)
        axes[2].set_title("Image Reconstituée")
        axes[2].axis("off")

        plt.show()

    show_images(full_image, noisy_img, reconstructed_full_image)



if __name__ == '__main__':
    main_2()
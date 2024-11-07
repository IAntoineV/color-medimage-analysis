import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

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
        t = t-1
        if isinstance(t, int):
            t= torch.tensor([t], dtype=torch.int, device=self.device)
        t = t.view(-1,1,1,1)
        epsilon_theta = self.model(xt, t)
        rescaling_coef_eps_theta = (self.betas[t]/torch.sqrt(1-self.alpha_bars[t])).view(-1, 1, 1, 1)
        last_rescale = (1/torch.sqrt(self.alphas[t])).view(-1,1,1,1)
        # Eq (11)
        x_denoised = last_rescale*(xt - epsilon_theta * rescaling_coef_eps_theta)
        if sampling:
            z = torch.randn_like(x_denoised).to(self.device)
            sigma_t = torch.sqrt(self.betas[t]).view(-1, 1, 1, 1)
            x_denoised = x_denoised +  (t>0)* sigma_t * z
        return x_denoised, epsilon_theta

    @torch.no_grad()
    def sampling(self, shape=None, xT=None,T=None):
        """
        Generate a new image

        :param t:
        :param shape:
        :return:
        """
        self.model.eval()
        assert shape is not None or xT is not None, "No specific prior or noise shape given !"

        if T is None:
            T = self.timesteps
        if xT is None:
            xT = torch.randn(shape)
        xt = xT.to(self.device)
        for denoising_step in range(T):
            timestep = T - denoising_step -1
            xt,_ = self.backward_diffusion(xt, timestep, sampling=True)
            #xt = xt.clip(0,1)
        x0 = xt
        return x0

    def train(self, dataloader, lr=1e-4, num_epochs=10, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        mse_loss = nn.MSELoss()
        device = self.device
        for epoch in range(num_epochs):
            i=0
            for images in dataloader():
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



class SmallUNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        super(SmallUNet, self).__init__()

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

    def forward(self, x, t):
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

def main():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import tqdm
    from unet import Unet
    # Set up data loader
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1),
    transforms.Resize((64,64))])
    #train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
    train_dataset = datasets.CelebA(root="data", transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    def gene():
        for elt in tqdm.tqdm(train_loader):
            yield elt[0]
    # Initialize diffusion process and model
    timesteps = 1000
    t_reconst=200
    num_epochs = 10
    device = torch.device("cuda")
    model = Unet(channels=64).to(device)
    nb_params = sum([elt.numel() for elt in model.parameters()])
    print(f"nb params : {nb_params}")
    diffusion = Diffusion(model, timesteps=timesteps, device=device)

    full_image = next(gene())[0].unsqueeze(0)  # Charger une image
    # Train model
    diffusion.train(gene, num_epochs=num_epochs)
    shape = full_image.shape
    img_gen = diffusion.sampling(shape)
    plt.imshow(np.clip(img_gen[0].permute(2,1,0).detach().cpu().numpy(), 0, 1))
    plt.show()



    noisy_img = full_image.clone()
    # noisy_img[0, :, :] = 0
    noisy_img, _ = diffusion.forward_diffusion(noisy_img, t_reconst)
    noisy_img = noisy_img.squeeze(0)
    reconstructed_full_image = diffusion.sampling(shape=None, xT=noisy_img, T=t_reconst).squeeze(0)
    # reconstructed_full_image = diffusion.sampling(shape = noisy_img.shape, T=t_reconst).squeeze(0).clip(0., 1.)
    # reconstructed_full_image = diffusion.sampling(xT=reconstructed_full_image, T=20).squeeze(0).clip(0., 1.)
    # Afficher les résultats


    show_images((1 + full_image[0]) / 2, (1 + noisy_img) / 2, (1 + reconstructed_full_image) / 2)

    torch.save(model.state_dict(), "model_latest.pt")

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
    from unet import Unet
    np.random.seed(42)
    torch.manual_seed(42)
    # Chargement du dataset avec extraction de patchs
    class HistoLiverPatchDataset(Dataset):

        def __init__(self, image_dir, nb_steps=1000, patch_size=64, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.patch_size = patch_size
            self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if
                           img.endswith('.jpg') or img.endswith('.png')]

            self.np_images = [Image.open(img_path).convert("RGB") for img_path in self.images]
            self.nb_img = len(self.images)
            self.nb_steps = nb_steps
        def __len__(self):
            return self.nb_steps

        def __getitem__(self, idx):
            # Randomly select an image from the list
            image_idx = np.random.choice(range(self.nb_img))
            image = self.np_images[image_idx]
            # Convert to tensor temporarily to get width and height
            image_tensor = 2*transforms.ToTensor()(image)-1
            _, img_height, img_width = image_tensor.shape

            # Randomly select the top-left corner of the patch
            max_row = img_height - self.patch_size
            max_col = img_width - self.patch_size
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)

            # Crop the patch from the image
            patch = image_tensor[:, row:row + self.patch_size, col:col + self.patch_size]

            return patch

    # Définir les transformations pour les images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Charger le dataset de patchs
    image_dir = '../../dataset/data/liver_HE'  # Assurez-vous que ce dossier contient vos images
    patch_size = 64
    batch_size = 64
    nb_patch_per_epoch = 6400
    dataset = HistoLiverPatchDataset(nb_steps= nb_patch_per_epoch ,image_dir=image_dir, patch_size=patch_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    timesteps = 1000
    t_reconst = 100
    device = torch.device("cuda")
    model = Unet().to(device)
    nb_params = sum([elt.numel() for elt in model.parameters()])
    print(f"nb params : {nb_params}")
    diffusion = Diffusion(model, timesteps=timesteps, beta_end=2e-2,device=device)
    diffusion.train(dataloader, lr=1e-3, num_epochs=50)

    # Chargement d'une image complète et affichage des résultats
    full_image = dataset[0][:,:patch_size,:patch_size]  # Charger une image
    noisy_img = full_image.clone()
    #noisy_img[0, :, :] = 0
    noisy_img, _ = diffusion.forward_diffusion(noisy_img, t_reconst)
    noisy_img = noisy_img.squeeze(0)
    reconstructed_full_image = diffusion.sampling(shape=None,xT=noisy_img, T=t_reconst).squeeze(0)
    #reconstructed_full_image = diffusion.sampling(shape = noisy_img.shape, T=t_reconst).squeeze(0).clip(0., 1.)
    #reconstructed_full_image = diffusion.sampling(xT=reconstructed_full_image, T=20).squeeze(0).clip(0., 1.)
    # Afficher les résultats


    show_images((1+full_image)/2, (1+noisy_img)/2, (1+reconstructed_full_image)/2)



if __name__ == '__main__':
    main()
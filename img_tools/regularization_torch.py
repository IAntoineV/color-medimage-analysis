
import torch

def total_variation_smoothing(H : torch.Tensor, EPS=1e-8):
    """
    Compute the Smoothing Total Variation (STV) of an image tensor H.

    Parameters:
    - H: torch.Tensor of shape (n,c,h,w) where n is the batch size, c is the number of channels,
         h is the height, and w is the width of the images.
    - EPS: numerical stability constant.

    Returns:
    - stv: Total Variation computed across all channels.
    """
    # Ensure H is a 4D tensor (N, C, H, W)
    if H.ndim != 4:
        raise ValueError("Input tensor H must be 4D (N, C, H, W).")

    n,c,h,w = H.shape

    DvH = H[:, :, 1:, :] - H[:, :, :-1, :]  # Vertical differences

    DhH = H[:, :, :, 1:] - H[:, :, :, :-1]  # Horizontal differences
    DvH_squared = DvH.pow(2) # (n,c,h-1,w)
    DhH_squared = DhH.pow(2) # (n,c,h,w-1)

    DvH_squared = DvH_squared[:,:,:,:-1] # Take down last line where we cannot compute the other directional gradient
    DhH_squared = DhH_squared[:,:,:-1,:] # Take down last line where we cannot compute the other directional gradient

    # shapes (n,c,h-1,w-1)

    # Sum over pixels
    stv_per_channel =  torch.sqrt(DvH_squared + DhH_squared+ EPS**2).sum(dim=(2,3)) # (n,c,)

    # Sum over channels
    stv_total = stv_per_channel.sum(dim=1)  # (n,)

    return stv_total



class Regularizer:



    @staticmethod
    def stv(H):


import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from matplotlib.figure import Figure
from .compute_diagnostics import velocity_from_vorticity, compute_spectra



def _to_rgb_minmax(image_2d: torch.Tensor) -> torch.Tensor:
    """Convert single-channel 2D field to 3-channel RGB with per-image min-max normalization.
    
    Args:
        image_2d: (H, W) tensor
        
    Returns:
        (3, H, W) tensor with RGB channels
    """
    img = image_2d.detach().float()
    min_val = torch.amin(img)
    max_val = torch.amax(img)
    if torch.isfinite(min_val) and torch.isfinite(max_val) and (max_val > min_val):
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)
    return img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)


def fig_to_tensorboard_image(fig: Figure) -> torch.Tensor:
    """Convert matplotlib figure to tensor for TensorBoard.
    
    Args:
        fig: matplotlib Figure object
        
    Returns:
        (3, H, W) tensor with RGB channels, values in [0, 1]
    """
    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    
    # Close figure to free memory
    plt.close(fig)
    
    # Convert to tensor and normalize to [0, 1]
    if len(img_array.shape) == 3:
        # RGB image
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        # Ensure 3 channels (RGB)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3]  # Remove alpha channel if present
        elif img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)  # Convert grayscale to RGB
    else:
        # Grayscale image
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)  # Convert to RGB
    
    return img_tensor


def log_tensorboard_images_and_spectra(
    writer,
    pred_denorm: torch.Tensor,
    target_denorm: torch.Tensor,
    epoch: int,
    form: str,
    model_name: str,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
    t_idx: int = 0
):
    """Log prediction, target, error images and energy/enstrophy spectra to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        pred_denorm: Denormalized predictions, shape (B, H, W, C)
        target_denorm: Denormalized targets, shape (B, H, W, C)
        epoch: Current epoch number
        form: Data form ('vorticity' or 'velocity')
        model_name: Name of the model (for plot labels)
        Lx: Domain size in x direction (default: 2*pi)
        Ly: Domain size in y direction (default: 2*pi)
    """
    B, H, W, C = pred_denorm.shape
    
    # Define channel names based on form
    if form == 'vorticity':
        channel_names = ['vorticity']
    elif form == 'velocity':
        channel_names = ['pressure', 'velocity_x', 'velocity_y']
    else:
        channel_names = [f'channel_{i}' for i in range(C)]
    
    # Log images for all output channels
    for channel_idx in range(C):
        # Use descriptive name if available, otherwise use channel index
        if channel_idx < len(channel_names):
            channel_name = channel_names[channel_idx]
        else:
            channel_name = f"channel_{channel_idx}"
        
        # Extract first batch, first time step
        pred_img = _to_rgb_minmax(pred_denorm[0, :, :, channel_idx])
        target_img = _to_rgb_minmax(target_denorm[0, :, :, channel_idx])
        error_img = _to_rgb_minmax(pred_denorm[0, :, :, channel_idx] - target_denorm[0, :, :, channel_idx])
        
        writer.add_image(f"pred/{channel_name}/t{t_idx}", pred_img, epoch)
        writer.add_image(f"target/{channel_name}/t{t_idx}", target_img, epoch)
        writer.add_image(f"error/{channel_name}/t{t_idx}", error_img, epoch)
    
    # Compute and plot energy and enstrophy spectra
    # Supports both vorticity and velocity forms
    if C == 1:
        try:            
            # Extract data for pred and target
            # Shape: (B, H, W, T, C) -> extract first batch, first time step
            pred_batch = pred_denorm[0].detach().cpu().numpy()  # (H, W, C)
            target_batch = target_denorm[0].detach().cpu().numpy()  # (H, W, C)
            
            # Get velocity components based on form
            if form == 'vorticity':

                ux_pred, uy_pred = velocity_from_vorticity(torch.from_numpy(pred_batch[..., 0]))
                ux_target, uy_target = velocity_from_vorticity(torch.from_numpy(target_batch[..., 0]))
            else:
                # Cannot compute spectra for this form/channel combination
                return
            
            # Compute spectra for prediction and target
            k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)
            _, Ek_target, Zk_target = compute_spectra(ux_target, uy_target, Lx, Ly)
            
            # Create energy spectrum plot
            fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
            k_nyquist = int((np.pi * H) // Lx)
            start_truth = 1
            ax_energy.loglog(k_bins[start_truth:k_nyquist], Ek_target[start_truth:k_nyquist], 
                          'X--', markersize=1, label='Ground Truth', linewidth=1, color='black')
            ax_energy.loglog(k_bins[start_truth:k_nyquist], Ek_pred[start_truth:k_nyquist], 
                          'o-', markersize=1, label=f'{model_name} Prediction', linewidth=1, color='blue')
            ax_energy.set_xlabel('Wavenumber', fontsize=14)
            ax_energy.set_ylabel('Energy', fontsize=14)
            ax_energy.set_title('Energy Spectrum', fontsize=14)
            ax_energy.legend(fontsize=12)
            ax_energy.grid(True)
            plt.tight_layout()
            
            # Convert to tensor and add to TensorBoard
            energy_img = fig_to_tensorboard_image(fig_energy)
            writer.add_image("spectra/energy_spectrum/t{t_idx}", energy_img, epoch)
            
            # Create enstrophy spectrum plot
            fig_enstrophy, ax_enstrophy = plt.subplots(figsize=(10, 6))
            ax_enstrophy.loglog(k_bins[start_truth:k_nyquist], Zk_target[start_truth:k_nyquist], 
                          'X-', markersize=2, label='Ground Truth', linewidth=2, color='black')
            ax_enstrophy.loglog(k_bins[start_truth:k_nyquist], Zk_pred[start_truth:k_nyquist], 
                          'o-', markersize=2, label=f'{model_name} Prediction', linewidth=2, color='blue')
            ax_enstrophy.set_xlabel('Wavenumber', fontsize=14)
            ax_enstrophy.set_ylabel('Enstrophy', fontsize=14)
            ax_enstrophy.set_title('Enstrophy Spectrum', fontsize=14)
            ax_enstrophy.legend(fontsize=12)
            ax_enstrophy.grid(True)
            plt.tight_layout()
            
            # Convert to tensor and add to TensorBoard
            enstrophy_img = fig_to_tensorboard_image(fig_enstrophy)
            writer.add_image("spectra/enstrophy_spectrum/t{t_idx}", enstrophy_img, epoch)
        except Exception as e:
            print(f"Warning: Failed to compute energy/enstrophy spectra: {e}")
            import traceback
            traceback.print_exc()
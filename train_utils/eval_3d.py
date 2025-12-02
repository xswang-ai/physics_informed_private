import math
import os

import torch
import torch.nn.functional as F

from tqdm import tqdm
from timeit import default_timer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .losses import LpLoss, PINO_loss3d
from .compute_diagnostics import compute_spectra_torch

try:
    import wandb
except ImportError:
    wandb = None


def eval_ns(model,  # model
            loader,  # dataset instance
            dataloader,  # dataloader
            forcing,  # forcing
            config,  # configuration dict
            device,  # device id
            log=False,
            project='PINO-default',
            group='FDM',
            tags=['Nan'],
            use_tqdm=True):
    '''
    Evaluate the model for Navier Stokes equation
    '''
    if wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))
    # data parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']
    # eval settings
    batch_size = config['test']['batchsize']

    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    loss_dict = {'f_error': 0.0,
                 'test_l2': 0.0}
    start_time = default_timer()
    example_pred = None
    example_truth = None
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5)
            out = out[..., :-5]
            x = x[:, :, :, 0, -1]
            loss_l2 = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
            loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)

            loss_dict['f_error'] += loss_f
            loss_dict['test_l2'] += loss_l2
            if example_pred is None:
                example_pred = out[0].detach().cpu()
                example_truth = y[0].detach().cpu()
            if device == 0 and use_tqdm:
                pbar.set_description(
                    (
                        f'Train f error: {loss_f.item():.5f}; Test l2 error: {loss_l2.item():.5f}'
                    )
                )
    end_time = default_timer()
    test_l2 = loss_dict['test_l2'].item() / len(dataloader)
    loss_f = loss_dict['f_error'].item() / len(dataloader)
    print(f'==Averaged relative L2 error is: {test_l2}==\n'
          f'==Averaged equation error is: {loss_f}==')
    print(f'Time cost: {end_time - start_time} s')

    if device == 0 and example_pred is not None:
        plot_dir = config.get('log', {}).get('plot_dir', '.')
        os.makedirs(plot_dir, exist_ok=True)

        # Use the last predicted time slice for quick visualization
        t_idx = example_pred.shape[-1] - 1
        pred_frame = example_pred[..., t_idx]
        truth_frame = example_truth[..., t_idx]
        err_frame = pred_frame - truth_frame

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ['Truth', 'Prediction', 'Error']
        data_to_plot = [truth_frame, pred_frame, err_frame]
        for ax, title, data in zip(axes, titles, data_to_plot):
            im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        pred_plot_path = os.path.join(plot_dir, 'ns_prediction.png')
        fig.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        def _velocity_from_vorticity(w_slice: torch.Tensor):
            """Compute velocity field from vorticity for spectrum calculation."""
            n = w_slice.shape[0]
            k_max = n // 2
            device_local = w_slice.device
            freq = torch.cat(
                (
                    torch.arange(0, k_max, device=device_local),
                    torch.arange(-k_max, 0, device=device_local),
                )
            )
            kx = freq.view(-1, 1).repeat(1, n)
            ky = freq.view(1, -1).repeat(n, 1)
            lap = kx ** 2 + ky ** 2
            lap[0, 0] = 1.0
            w_hat = torch.fft.fft2(w_slice)
            psi_hat = w_hat / lap
            ux_hat = 1j * ky * psi_hat
            uy_hat = -1j * kx * psi_hat
            ux = torch.fft.ifft2(ux_hat).real
            uy = torch.fft.ifft2(uy_hat).real
            return ux, uy

        try:
            ux_pred, uy_pred = _velocity_from_vorticity(pred_frame.float())
            ux_true, uy_true = _velocity_from_vorticity(truth_frame.float())
            k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
            _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)

            k_np = k_bins.cpu().numpy()
            Ek_pred_np = Ek_pred.cpu().numpy()
            Ek_true_np = Ek_true.cpu().numpy()

            # valid_mask = k_np > 0
            valid_mask = range(1, 33)
            fig_spec, ax_spec = plt.subplots(1, 1, figsize=(6, 4))
            ax_spec.loglog(k_np[valid_mask], Ek_true_np[valid_mask], label='Truth', linewidth=1)
            ax_spec.loglog(k_np[valid_mask], Ek_pred_np[valid_mask], '--', label='Prediction', linewidth=1)
            ax_spec.set_xlabel('Wavenumber k')
            ax_spec.set_ylabel('Energy E(k)')
            ax_spec.set_title('Spectral Energy Comparison')
            ax_spec.grid(True, which='both', alpha=0.3)
            ax_spec.legend()
            spec_plot_path = os.path.join(plot_dir, 'ns_spectral_energy.png')
            fig_spec.savefig(spec_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_spec)
        except Exception as exc:  # noqa: BLE001
            print(f'Warning: failed to create spectral energy plot: {exc}')

    if device == 0:
        if wandb and log:
            wandb.log(
                {
                    'Train f error': loss_f,
                    'Test L2 error': test_l2,
                }
            )
            run.finish()

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
from .compute_diagnostics import compute_spectra_torch, velocity_from_vorticity

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
            use_tqdm=True,
            max_time_steps=None):
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
            print("x shape: ", x.shape, "y shape: ", y.shape)
            # truncate x and y here if the time steps is greater the model.
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in)
            print("out shape: ", out.shape)
            out = out.reshape(batch_size, S, S, -1)
            y = y[..., :out.shape[-1]]
            if max_time_steps is not None:
                t_keep = min(max_time_steps, out.shape[-1], y.shape[-1])
                out = out[..., :t_keep]
                y = y[..., :t_keep]
                T = t_keep
            x = x[:, :, :, 0, -1]
            loss_l2 = myloss(out.view(batch_size, S, S, -1), y.view(batch_size, S, S, -1))
            # loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)
            loss_f = torch.tensor(0.0, device=device)
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
        print("plot_dir: ", plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        pred_dir = os.path.join(plot_dir, 'saved_plots','predictions')
        spec_dir = os.path.join(plot_dir, 'saved_plots','spectral_energy')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        time_indices = torch.arange(0, example_pred.shape[-1], 5)
        for t_raw in time_indices:
            if t_raw < 0:
                t_idx = example_pred.shape[-1] + t_raw
            else:
                t_idx = t_raw
            if t_idx < 0 or t_idx >= example_pred.shape[-1]:
                continue

            pred_frame = example_pred[..., t_idx]
            truth_frame = example_truth[..., t_idx]
            err_frame = pred_frame - truth_frame

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            titles = ['Truth', 'Prediction', 'Error']
            data_to_plot = [truth_frame, pred_frame, err_frame]

            truth_min = truth_frame.min().item()
            truth_max = truth_frame.max().item()
            abs_lim = max(abs(truth_min), abs(truth_max))
            vmin = -abs_lim
            vmax = abs_lim
            for ax, title, data in zip(axes, titles, data_to_plot):
                if title in ['Truth', 'Prediction']:
                    im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
                else:
                    err_abs = max(abs(data.min().item()), abs(data.max().item()), 1e-8)
                    im = ax.imshow(data.numpy(), cmap='RdBu_r', origin='lower', vmin=-err_abs, vmax=err_abs)
                ax.set_title(f'{title} (T={t_raw})')
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            pred_plot_path = os.path.join(pred_dir, f'ns_prediction_t{t_raw}.png')
            fig.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            try:
                ux_pred, uy_pred = velocity_from_vorticity(pred_frame.float())
                ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
                k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
                _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)

                k_np = k_bins.cpu().numpy()
                Ek_pred_np = Ek_pred.cpu().numpy()
                Ek_true_np = Ek_true.cpu().numpy()

                valid_mask = range(1, 33)
                fig_spec, ax_spec = plt.subplots(1, 1, figsize=(6, 4))
                ax_spec.loglog(k_np[valid_mask], Ek_true_np[valid_mask], label='Truth', linewidth=1)
                ax_spec.loglog(k_np[valid_mask], Ek_pred_np[valid_mask], '--', label='Prediction', linewidth=1)
                ax_spec.set_xlabel('Wavenumber k')
                ax_spec.set_ylabel('Energy E(k)')
                ax_spec.set_title(f'Spectral Energy Comparison (T={t_raw})')
                ax_spec.grid(True, which='both', alpha=0.3)
                ax_spec.legend()
                spec_plot_path = os.path.join(spec_dir, f'ns_spectral_energy_t{t_raw}.png')
                fig_spec.savefig(spec_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig_spec)
            except Exception as exc:  # noqa: BLE001
                print(f'Warning: failed to create spectral energy plot at T={t_raw}: {exc}')

    if device == 0:
        if wandb and log:
            wandb.log(
                {
                    'Train f error': loss_f,
                    'Test L2 error': test_l2,
                }
            )
            run.finish()

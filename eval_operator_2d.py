import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

from baselines import data
from models import FNO3d, FNO2d
from models.wavelet_transform_exploration import WaveletTransformer2D, InnerWaveletTransformer2D, MultiscaleWaveletTransformer2D
from models.hfs import ResUNet
from models.wno import WNO2d
from models.saot import SAOTModel

from train_utils.losses import LpLoss
from train_utils.utils import get_grid3d, torch2dgrid
from train_utils.datasets import NSLoader2D
from train_utils.compute_diagnostics import compute_scalar_diagnostics, compute_spectra_torch
from train_utils.eval_3d import velocity_from_vorticity
from train_utils.eval_3d import velocity_from_vorticity
import os
import math
import matplotlib.pyplot as plt

def load_ns_sequences(data_config):
    """Load full (N, X, Y, T) sequences for evaluation."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config['nx']
    nt = data_config['nt']
    t_interval = data_config.get('time_interval', 1.0)
    datapath1 = data_config['datapath']

    S = nx // sub
    T = int(nt * t_interval) // sub_t + 1

    data1 = np.load(datapath1)
    data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]

    if t_interval == 0.5:
        # subselect time to 1s 
        # data1 = NSLoader2D.extract(data1)
        sub_t = 1//t_interval
        data1 = data1[..., ::sub_t, ...]
        
    part1 = data1.permute(0, 2, 3, 1)  # (N, X, Y, T)
    data = part1
    print("data shape: ", data.shape)

    offset = data_config.get('offset', 0)
    n_sample = data_config.get('n_sample', data_config.get('n_samples', data_config.get('total_num', data.shape[0])))
    end = min(data.shape[0], offset + n_sample)
    data = data[offset:end]
    print("final data shape: ", data.shape)
    exit(-1)
    return data, S, T


def autoregressive_eval(model, sequences, device):
    """Run autoregressive rollout on full sequences."""
    lploss = LpLoss(size_average=True)
    model.eval()
    S = sequences.shape[1]
    T = sequences.shape[-1]
    grid = torch2dgrid(S, S).to(device).unsqueeze(0)  # 1 x S x S x 2
    total = 0.0
    batches = 0
    example = {'truth': None, 'pred': None}
    loader = DataLoader(TensorDataset(sequences), batch_size=1, shuffle=False)
    with torch.no_grad():
        for (seq,) in loader:
            seq = seq.to(device)  # (1, S, S, T)
            preds = []  # predicted rollout
            prev = seq[..., 0]  # initial condition
            for t in range(T - 1):
                x_in = torch.cat((prev.unsqueeze(-1), grid.expand(prev.shape[0], -1, -1, -1)), dim=-1)
                pred = model(x_in)
                if pred.dim() == 5:
                    pred = pred.squeeze(-2)
                if pred.dim() == 4:
                    pred = pred.squeeze(-1)
                preds.append(pred)
                prev = pred
            pred_seq = torch.stack(preds, dim=-1)       # (1, S, S, T-1)
            truth_seq = seq[..., 1:]                    # align with predictions
            total += lploss(pred_seq.view(1, S, S, T - 1),
                            truth_seq.view(1, S, S, T - 1)).item()
            batches += 1
            if example['truth'] is None:
                example['truth'] = truth_seq.detach().cpu()
                example['pred'] = pred_seq.detach().cpu()
    return total / max(1, batches), example



def main():
    parser = ArgumentParser(description='Evaluate 2D operator autoregressively')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    model_cfg = config['model']
    sequences, S, T = load_ns_sequences(data_config)

    model_name = model_cfg.get('name', 'fno2d').lower()
    
    if model_name == 'hfs':
        model = ResUNet(in_c=3,
                        out_c=1,
                        target_params=model_cfg.get('target_params', 'medium'),
                        device=device).to(device)
    elif model_name in ['wno', 'wno2d']:
        dummy = torch.zeros(1, 1, S_data, S_data, device=device)
        model = WNO2d(in_channels=model_cfg.get('in_chans', 3),
                      out_channels=model_cfg.get('out_chans', 1),
                      width=model_cfg.get('width', 64),
                      level=model_cfg.get('level', 3),
                      dummy_data=dummy).to(device)
    elif model_name in ['wavelet', 'wavelet2d', 'wavelet_transformer2d']:
        patch_size = model_cfg.get('patch_size', None)
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        model = WaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            in_chans=model_cfg.get('in_chans', 3),
            out_chans=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', 128),
            depth=model_cfg.get('depth', 4),
            patch_size=patch_size,
            patch_stride=model_cfg.get('patch_stride', 2),
            learnable_scaling_factor=model_cfg.get('learnable_scaling_factor', False),
        ).to(device)
    elif model_name in ['inner_wavelet', 'inner_wavelet2d', 'inner_wavelet_transformer2d']:
        
        model = InnerWaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),  # expecting u with grid concatenated
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', 128),
            n_layers=model_cfg.get('n_layers', 5),
            patch_size= model_cfg.get('patch_size', None),
        ).to(device)
    elif model_name in ['multiscale_wavelet', 'multiscale_wavelet2d', 'multiscale_wavelet_transformer2d']:
        model = MultiscaleWaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', 3),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', 128),
            n_layers=model_cfg.get('n_layers', 5),
            patch_size= model_cfg.get('patch_size', None),
        ).to(device)
    elif model_name in ['saot', 'saot2d']:
        model = SAOTModel(space_dim=model_cfg.get('space_dim', 2),
                        n_layers=model_cfg.get('n_layers', 3),
                        n_hidden=model_cfg.get('n_hidden', 64)  ,
                        dropout=model_cfg.get('dropout', 0.0),
                        n_head=model_cfg.get('n_head', 4),
                        Time_Input=model_cfg.get('Time_Input', False),
                        mlp_ratio=model_cfg.get('mlp_ratio', 1),
                        fun_dim=model_cfg.get('fun_dim', 1),
                        out_dim=model_cfg.get('out_dim', 1),
                        H = S_data,
                        W = S_data,
                        slice_num=model_cfg.get('slice_num', 32),
                        ref=model_cfg.get('ref', 8),
                        unified_pos=model_cfg.get('unified_pos', 0),
                        is_filter=model_cfg.get('is_filter', True)).to(device)
    else:
        model = FNO2d(modes1=model_cfg['modes1'],
                      modes2=model_cfg['modes2'],
                      fc_dim=model_cfg['fc_dim'],
                      layers=model_cfg['layers'],
                      act=model_cfg['act'],
                    #   pad_ratio=model_cfg.get('pad_ratio', [0., 0.])
                      ).to(device)
    print('model structure: ', model)

    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))

    base_ckpt_root = '/scratch3/wan410/operator_learning_model/pino_ns2d/checkpoints'
    train_cfg = config.get('train', {})
    default_ckpt_path = os.path.join(
        base_ckpt_root,
        train_cfg.get('save_dir', 'default'),
        train_cfg.get('save_name', 'model.pt')
    )
    ckpt_path = config.get('test', {}).get('ckpt', default_ckpt_path)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Weights loaded from {ckpt_path}')
    else:
        print(f'Checkpoint not found at {ckpt_path}; evaluating with randomly initialized weights.')

    print(f'Evaluating on {sequences.shape[0]} samples at resolution {S}x{S} for {T} steps.')
    l2, example = autoregressive_eval(model, sequences, device)
    print(f'Relative L2 over rollout: {l2:.6f}')

    # Save prediction and energy plots for the first example
    if example['truth'] is not None:
        plot_dir = config.get('log', {}).get(
            'plot_dir',
            os.path.join(base_ckpt_root, train_cfg.get('save_dir', 'default'), 'eval_plots')
        )
        pred_dir = os.path.join(plot_dir, 'saved_plots', 'predictions')
        spec_dir = os.path.join(plot_dir, 'saved_plots', 'energy')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        truth = example['truth'][0]  # (S, S, T-1)
        pred = example['pred'][0]
        T_pred = pred.shape[-1]
        time_indices = range(0, T_pred, 5)
        for t_raw in time_indices:
            pred_frame = pred[..., t_raw]
            truth_frame = truth[..., t_raw]
            err_frame = pred_frame - truth_frame
            truth_min = truth_frame.min().item()
            truth_max = truth_frame.max().item()
            abs_lim = max(abs(truth_min), abs(truth_max))
            vmin = -abs_lim
            vmax = abs_lim

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            titles = ['Truth', 'Prediction', 'Error']
            data_to_plot = [truth_frame, pred_frame, err_frame]
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

            # Spectral energy comparison
            try:
                ux_pred, uy_pred = velocity_from_vorticity(pred_frame.float())
                ux_true, uy_true = velocity_from_vorticity(truth_frame.float())
                k_bins, Ek_pred = compute_spectra_torch(ux_pred, uy_pred, 2 * math.pi, 2 * math.pi)
                _, Ek_true = compute_spectra_torch(ux_true, uy_true, 2 * math.pi, 2 * math.pi)

                k_np = k_bins.cpu().numpy()
                Ek_pred_np = Ek_pred.cpu().numpy()
                Ek_true_np = Ek_true.cpu().numpy()

                valid_mask = range(1, min(len(k_np), S // 2))
                fig_spec, ax_spec = plt.subplots(1, 1, figsize=(6, 4))
                ax_spec.loglog(k_np[valid_mask], Ek_true_np[valid_mask], label='Truth', linewidth=1)
                ax_spec.loglog(k_np[valid_mask], Ek_pred_np[valid_mask], '--', label='Prediction', linewidth=1)
                ax_spec.set_xlabel('Wavenumber k')
                ax_spec.set_ylabel('Energy E(k)')
                ax_spec.set_title(f'Spectral Energy (T={t_raw})')
                ax_spec.grid(True, which='both', alpha=0.3)
                ax_spec.legend()
                spec_plot_path = os.path.join(spec_dir, f'ns_spectral_energy_t{t_raw}.png')
                fig_spec.savefig(spec_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig_spec)
            except Exception as exc:  # noqa: BLE001
                print(f'Warning: failed to create spectral energy plot at T={t_raw}: {exc}')


# if __name__ == '__main__':
#     main()

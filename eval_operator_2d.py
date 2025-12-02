import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

from models import FNO2d
from models.hfs import ResUNet
from train_utils.losses import LpLoss
from train_utils.utils import get_grid3d, torch2dgrid
from train_utils.datasets import NSLoader2D


def load_ns_sequences(data_config):
    """Load full (N, X, Y, T) sequences for evaluation."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config['nx']
    nt = data_config['nt']
    t_interval = data_config.get('time_interval', 1.0)
    datapath1 = data_config['datapath']
    datapath2 = data_config.get('datapath2', None)

    S = nx // sub
    T = int(nt * t_interval) // sub_t + 1

    data1 = np.load(datapath1)
    data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
    if datapath2 is not None:
        data2 = np.load(datapath2)
        data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
    if t_interval == 0.5:
        data1 = NSLoader2D.extract(data1)
        if datapath2 is not None:
            data2 = NSLoader2D.extract(data2)
    part1 = data1.permute(0, 2, 3, 1)  # (N, X, Y, T)
    if datapath2 is not None:
        part2 = data2.permute(0, 2, 3, 1)
        data = torch.cat((part1, part2), dim=0)
    else:
        data = part1

    offset = data_config.get('offset', 0)
    n_sample = data_config.get('n_sample', data_config.get('n_samples', data_config.get('total_num', data.shape[0])))
    end = min(data.shape[0], offset + n_sample)
    data = data[offset:end]
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
    loader = DataLoader(TensorDataset(sequences), batch_size=1, shuffle=False)
    with torch.no_grad():
        for (seq,) in loader:
            seq = seq.to(device)  # (1, S, S, T)
            preds = [seq[..., 0]]  # start from t0 ground truth
            prev = seq[..., 0]
            for t in range(T - 1):
                x_in = torch.cat((prev.unsqueeze(-1), grid.expand(prev.shape[0], -1, -1, -1)), dim=-1)
                pred = model(x_in)
                if pred.dim() == 5:
                    pred = pred.squeeze(-2)
                if pred.dim() == 4:
                    pred = pred.squeeze(-1)
                preds.append(pred)
                prev = pred
            pred_seq = torch.stack(preds, dim=-1)
            total += lploss(pred_seq.view(1, S, S, T), seq.view(1, S, S, T)).item()
            batches += 1
    return total / max(1, batches)


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
    else:
        model = FNO2d(modes1=model_cfg['modes1'],
                      modes2=model_cfg['modes2'],
                      fc_dim=model_cfg['fc_dim'],
                      layers=model_cfg['layers'],
                      act=model_cfg['act'],
                      pad_ratio=model_cfg.get('pad_ratio', [0., 0.])).to(device)

    if 'ckpt' in config.get('test', {}):
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Weights loaded from {ckpt_path}')

    print(f'Evaluating on {sequences.shape[0]} samples at resolution {S}x{S} for {T} steps.')
    l2 = autoregressive_eval(model, sequences, device)
    print(f'Relative L2 over rollout: {l2:.6f}')


if __name__ == '__main__':
    main()

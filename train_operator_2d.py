import os
import yaml
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from solver.random_fields import GaussianRF
from train_utils import Adam
from train_utils.plot_utils import log_tensorboard_images_and_spectra
from train_utils.datasets import NSLoader, online_loader, DarcyFlow, DarcyCombo, NSLoader2D
from train_utils.losses import LpLoss
from train_utils.utils import get_grid3d, torch2dgrid, save_checkpoint
from models import FNO3d, FNO2d
from models.wavelet_transform_exploration import WaveletTransformer2D, InnerWaveletTransformer2D
from models.hfs import ResUNet
from models.wno import WNO2d
from tqdm import tqdm


def evaluate_3d(model, test_loader, device):
    """Run a quick L2 evaluation on a held-out set."""
    lploss = LpLoss(size_average=True)
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size, S, _, T, _ = x.shape
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5)
            out = out[..., :-5]
            total += lploss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T)).item()
            batches += 1
    if batches == 0:
        return None
    return total / batches


def evaluate_step_ahead(model, test_loader, device, grid):
    """Evaluate one-step prediction u_t -> u_{t+1}."""
    lploss = LpLoss(size_average=True)

    model.eval()
    total = 0.0
    batches = 0
    pred_plot = None
    target_plot = None
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch = x.shape[0]
            x_in = torch.cat((x.unsqueeze(-1), grid.expand(batch, -1, -1, -1)), dim=-1)
            pred = model(x_in)
            if pred.dim() == 5:
                pred = pred.squeeze(-2)
            if pred.dim() == 4:
                pred = pred.squeeze(-1)
            total += lploss(pred, y).item()
            if pred_plot is None:
                pred_plot = pred.clone()
                target_plot = y.clone()
            batches += 1
    if batches == 0:
        return None
    return total / batches, pred_plot, target_plot


def _get_base_dataset(ds):
    """Return the underlying dataset (unwrap Subset/DataLoader)."""
    if isinstance(ds, DataLoader):
        ds = ds.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_fixed_test_pair(model, test_source, grid, device, sample_idx=0, t_idx=0):
    """
    Grab a deterministic (x_t, x_{t+1}) pair from the test data without relying on
    the test loader's random timestep selection.
    """
    base_ds = _get_base_dataset(test_source)
    if not hasattr(base_ds, 'data'):
        return None, None
    data = base_ds.data
    if sample_idx >= data.shape[0]:
        sample_idx = data.shape[0] - 1
    max_t = data.shape[-1] - 1
    if max_t <= 0:
        return None, None
    t_idx = min(t_idx, max_t - 1)

    sample = data[sample_idx]
    x = sample[..., t_idx].to(device)
    y = sample[..., t_idx + 1].to(device)
    grid_b = grid.to(device)
    x_in = torch.cat((x.unsqueeze(0).unsqueeze(-1), grid_b), dim=-1)
    with torch.no_grad():
        pred = model(x_in)
        if pred.dim() == 5:
            pred = pred.squeeze(-2)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
    return pred, y.unsqueeze(0)


def train_step_ahead(model, train_loader, optimizer, scheduler, config, device, grid, test_loader=None, eval_step=100,save_step=1000, use_tqdm=True, writer=None, model_name='fno2d'):
    """Train on one-step pairs (u_t, u_{t+1})."""
    lploss = LpLoss(size_average=True)
    epochs = config['train']['epochs']
    grid = grid.to(device).unsqueeze(0)
    if use_tqdm:
        pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.1)
    else:
        pbar = range(epochs)
    for ep in pbar:
        model.train()
        running = 0.0
        batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch = x.shape[0]
            x_in = torch.cat((x.unsqueeze(-1), grid.expand(batch, -1, -1, -1)), dim=-1)
            pred = model(x_in)
            if pred.dim() == 5:
                pred = pred.squeeze(-2)
            if pred.dim() == 4:
                pred = pred.squeeze(-1)
            loss = lploss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1
        scheduler.step()
        avg = running / max(1, batches)
        print(f'Epoch {ep + 1}/{epochs}, train L2: {avg:.6f}')
        if writer is not None:
            writer.add_scalar('train/l2', avg, ep + 1)
        if use_tqdm:
            pbar.set_description((f'Train L2: {avg:.6f}'))

        if ep % eval_step == 0 and test_loader is not None:
            test_l2, _, _ = evaluate_step_ahead(model, test_loader, device, grid)
            print(f'Random test split relative L2: {test_l2:.6f}')
            if writer is not None:
                writer.add_scalar('eval/test_l2', test_l2, ep + 1)
                fixed_pred, fixed_target = get_fixed_test_pair(model, test_loader, grid, device, sample_idx=0, t_idx=0)
                if fixed_pred is not None:
                    log_tensorboard_images_and_spectra(writer,
                                                       fixed_pred.unsqueeze(-1),
                                                       fixed_target.unsqueeze(-1),
                                                       ep + 1,
                                                       'vorticity',
                                                       model_name)

        if ep % save_step == 0 and ep > 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{ep + 1}.pt'),
                            model, optimizer, scheduler)


def build_synthetic_dataset(data_config, n_samples, step_ahead=False):
    """Create a random dataset that mimics NSLoader/NSLoader2D output."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config.get('nx', 64)
    nt = data_config.get('nt', 64)
    time_scale = data_config.get('time_interval', 1.0)
    S = nx // sub
    T = int(nt * time_scale) // sub_t + 1

    if step_ahead:
        data = torch.rand(n_samples, S, S, T)

        class SyntheticStepDataset(Dataset):
            def __init__(self, arr):
                self.arr = arr
                self.max_t = arr.shape[-1] - 1

            def __len__(self):
                return self.arr.shape[0]

            def __getitem__(self, idx):
                sample = self.arr[idx]
                t = torch.randint(0, self.max_t, ()).item()
                return sample[..., t], sample[..., t + 1]

        return SyntheticStepDataset(data), S, 1

    a0 = torch.rand(n_samples, S, S, 1, 1)
    a_data = a0.repeat(1, 1, 1, T, 1)
    gridx, gridy, gridt = get_grid3d(S, T, time_scale=time_scale)
    a_data = torch.cat((
        gridx.repeat([n_samples, 1, 1, 1, 1]),
        gridy.repeat([n_samples, 1, 1, 1, 1]),
        gridt.repeat([n_samples, 1, 1, 1, 1]),
        a_data
    ), dim=-1)
    u_data = torch.rand(n_samples, S, S, T)
    return TensorDataset(a_data, u_data), S, T


def train_3d(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, T_data = build_synthetic_dataset(
            data_config, args.synthetic_samples, step_ahead=True)
    else:

        n_sample = data_config.get('n_sample', data_config.get('n_samples', data_config['total_num']))
        offset = data_config.get('offset', 0)
        full_dataset = NSLoader2D(datapath1=data_config['datapath'],
                                    nx=data_config['nx'], nt=data_config['nt'],
                                    sub=data_config['sub'], sub_t=data_config['sub_t'],
                                    N=data_config['total_num'],
                                    t_interval=data_config['time_interval'],
                                    n_samples=n_sample,
                                    offset=offset)
        S_data, T_data = full_dataset.S, 1
    if args.test_ratio > 0:
        test_size = max(1, int(len(full_dataset) * args.test_ratio))
        if len(full_dataset) - test_size <= 0:
            raise ValueError('test_ratio is too large; no samples left for training.')
        train_size = len(full_dataset) - test_size
        train_set, test_set = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.test_seed)
        )
        test_set.train = False # set test set to not train
        test_loader = DataLoader(test_set,
                                 batch_size=config['train']['batchsize'],
                                 shuffle=False)
    else:
        train_set = full_dataset
        test_loader = None

    train_loader = DataLoader(train_set,
                              batch_size=config['train']['batchsize'],
                              shuffle=data_config['shuffle'])
    # create model
    print("device: ", device)
    model_cfg = config['model']
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
        patch_size = model_cfg.get('patch_size', (4, 4))
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
            patch_size=model_cfg.get('patch_size', 4),
        ).to(device)
    else:
        model = FNO2d(modes1=model_cfg['modes1'],
                      modes2=model_cfg['modes2'],
                      fc_dim=model_cfg['fc_dim'],
                      layers=model_cfg['layers'],
                      act=model_cfg['act'],
                    #   pad_ratio=model_cfg.get('pad_ratio', [0., 0.])
                      ).to(device)
    print('model structure: ', model)
    
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    # create optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])


    save_dir = config['train']['save_dir'] if torch.cuda.is_available() else 'saved_models'
    tensorboard_dir = config['train'].get('tensorboard_dir')
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    grid = torch2dgrid(S_data, S_data)
    train_step_ahead(model,
                        train_loader,
                        optimizer,
                        scheduler,
                        config,
                        device,
                        grid,
                        test_loader=test_loader,
                        writer=writer,
                        model_name=model_name)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer, scheduler)
    if test_loader is not None:
        test_l2, _, _ = evaluate_step_ahead(model, test_loader, device, grid.to(device).unsqueeze(0))
        print(f'Random test split relative L2: {test_l2:.6f}')
        if writer is not None:
            writer.add_scalar('eval/test_l2', test_l2, config['train']['epochs'])
            fixed_pred, fixed_target = get_fixed_test_pair(model, test_loader, grid, device, sample_idx=0, t_idx=0)
            if fixed_pred is not None:
                log_tensorboard_images_and_spectra(writer,
                                                   fixed_pred.unsqueeze(-1),
                                                   fixed_target.unsqueeze(-1),
                                                   config['train']['epochs'],
                                                   'vorticity',
                                                   model_name)
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--test_ratio', type=float, default=0.0,
                        help='Hold out this fraction of samples for a random test split (3D only)')
    parser.add_argument('--test_seed', type=int, default=42,
                        help='Seed for the random test split')
    parser.add_argument('--synthetic_samples', type=int, default=0,
                        help='Use random synthetic data with this many samples to sanity-check the 3D pipeline')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    train_3d(args, config)

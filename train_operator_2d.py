import yaml
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset

from solver.random_fields import GaussianRF
from train_utils import Adam
from train_utils.datasets import NSLoader, online_loader, DarcyFlow, DarcyCombo, NSLoader2D
from train_utils.train_3d import mixed_train
from train_utils.train_2d import train_2d_operator
from train_utils.losses import LpLoss
from train_utils.utils import get_grid3d, torch2dgrid
from models import FNO3d, FNO2d
from models.hfs import ResUNet
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
    grid = grid.to(device).unsqueeze(0)
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
            batches += 1
    if batches == 0:
        return None
    return total / batches


def train_step_ahead(model, train_loader, optimizer, scheduler, config, device, grid, use_tqdm=True):
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

        if use_tqdm:
            pbar.set_description(
                (
                    f'Train L2: {avg:.6f}'
                )
            )


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
    step_ahead_mode = 'datapath2' not in data_config

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, T_data = build_synthetic_dataset(
            data_config, args.synthetic_samples, step_ahead=step_ahead_mode)
    else:
        if not step_ahead_mode:
            loader = NSLoader(datapath1=data_config['datapath'], datapath2=data_config['datapath2'],
                              nx=data_config['nx'], nt=data_config['nt'],
                              sub=data_config['sub'], sub_t=data_config['sub_t'],
                              N=data_config['total_num'],
                              t_interval=data_config['time_interval'])
            # Build dataset and (optionally) split out a random test set
            full_dataset = loader.make_dataset(data_config['n_sample'],
                                               start=data_config['offset'],
                                               train=True)
            S_data, T_data = loader.S, loader.T
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
        test_loader = DataLoader(test_set,
                                 batch_size=config['train']['batchsize'],
                                 shuffle=False)
    else:
        train_set = full_dataset
        test_loader = None

    train_loader = DataLoader(train_set,
                              batch_size=config['train']['batchsize'],
                              shuffle=data_config['shuffle'])
    if not step_ahead_mode:
        # prepare dataloader for training with only equations
        s2 = data_config.get('S2', S_data)
        t2 = data_config.get('T2', T_data)
        gr_sampler = GaussianRF(2, s2, 2 * math.pi, alpha=2.5, tau=7, device=device)
        a_loader = online_loader(gr_sampler,
                                 S=s2,
                                 T=t2,
                                 time_scale=data_config['time_interval'],
                                 batchsize=config['train']['batchsize'])
    # create model
    print(device)
    model_cfg = config['model']
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
    if step_ahead_mode:
        grid = torch2dgrid(S_data, S_data)
        train_step_ahead(model,
                         train_loader,
                         optimizer,
                         scheduler,
                         config,
                         device,
                         grid)
        if test_loader is not None:
            test_l2 = evaluate_step_ahead(model, test_loader, device, grid)
            print(f'Random test split relative L2: {test_l2:.6f}')
    else:
        mixed_train(model,
                    train_loader,
                    S_data, T_data,
                    a_loader,
                    s2, t2,
                    optimizer,
                    scheduler,
                    config,
                    device,
                    log=args.log,
                    project=config['log']['project'],
                    group=config['log']['group'])

        if test_loader is not None:
            test_l2 = evaluate_3d(model, test_loader, device)
            print(f'Random test split relative L2: {test_l2:.6f}')


def train_2d(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])
    
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    model_cfg = config['model']
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
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config, rank=0, log=args.log,
                      project=config['log']['project'],
                      group=config['log']['group'])


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

    if 'name' in config['data'] and config['data']['name'] == 'Darcy':
        train_2d(args, config)
    else:
        train_3d(args, config)

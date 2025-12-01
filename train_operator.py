import yaml
from argparse import ArgumentParser
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from solver.random_fields import GaussianRF
from train_utils import Adam
from train_utils.datasets import NSLoader, online_loader, DarcyFlow, DarcyCombo
from train_utils.train_3d import mixed_train
from train_utils.train_2d import train_2d_operator
from train_utils.losses import LpLoss
from train_utils.utils import get_grid3d
from models import FNO3d, FNO2d


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


def build_synthetic_dataset(data_config, n_samples):
    """Create a random (a, u) dataset that mimics NSLoader output."""
    sub = data_config.get('sub', 1)
    sub_t = data_config.get('sub_t', 1)
    nx = data_config.get('nx', 64)
    nt = data_config.get('nt', 64)
    time_scale = data_config.get('time_interval', 1.0)
    S = nx // sub
    T = int(nt * time_scale) // sub_t + 1

    a0 = torch.rand(n_samples, S, S, 1, 1)  # initial field
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
        full_dataset, S_data, T_data = build_synthetic_dataset(data_config, args.synthetic_samples)
    else:
        if 'datapath2' in data_config:
            loader = NSLoader(datapath1=data_config['datapath'], datapath2=data_config['datapath2'],
                              nx=data_config['nx'], nt=data_config['nt'],
                              sub=data_config['sub'], sub_t=data_config['sub_t'],
                              N=data_config['total_num'],
                              t_interval=data_config['time_interval'])
        else:
            loader = NSLoader(datapath1=data_config['datapath'],
                              nx=data_config['nx'], nt=data_config['nt'],
                              sub=data_config['sub'], sub_t=data_config['sub_t'],
                              N=data_config['total_num'],
                              t_interval=data_config['time_interval'])

        # Build dataset and (optionally) split out a random test set
        full_dataset = loader.make_dataset(data_config['n_sample'],
                                           start=data_config['offset'],
                                           train=True)
        S_data, T_data = loader.S, loader.T
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
    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act']).to(device)
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

    # dataset = DarcyFlow(data_config['datapath'],
                        # nx=data_config['nx'], sub=data_config['sub'],
                        # offset=data_config['offset'], num=data_config['n_sample'])

    dataset = DarcyCombo(datapath=data_config['datapath'], 
                         nx=data_config['nx'], 
                         sub=data_config['sub'], 
                         pde_sub=data_config['pde_sub'], 
                         num=data_config['n_samples'], 
                         offset=data_config['offset'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio']).to(device)
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

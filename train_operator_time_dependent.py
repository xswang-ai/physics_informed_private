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
from models.wavelet_transform_exploration import MSWT2DStable, MSWT2DStableSoftControl, MSWT2DStableNormalizedEnergy
from models.hfs import ResUNet
from models.wno import WNO2d
from models.saot import SAOTModel
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


def _unwrap_dataset_and_indices(ds):
    """
    Unwrap DataLoader/Subset nesting and return (base_dataset, indices).

    If ds is a Subset (possibly nested), indices map ds idx -> base_dataset idx.
    """
    if isinstance(ds, DataLoader):
        ds = ds.dataset
    indices = None
    while isinstance(ds, Subset):
        cur = list(ds.indices)
        if indices is None:
            indices = cur
        else:
            indices = [indices[i] for i in cur]
        ds = ds.dataset
    return ds, indices


class InitialConditionTimeDataset(Dataset):
    """
    Sample (x0, t, x_t) from trajectory data shaped (N, S, S, T).

    - x0 is always the initial condition at t=0.
    - t is a scalar (float32) per sample.
    - x_t is the ground-truth field at the sampled time index.
    """

    def __init__(
        self,
        source,
        time_sampling="random",
        fixed_t_idx=1,
        min_t_idx=1,
        time_encoding="normalized",
        time_interval=1.0,
    ):
        super().__init__()
        if isinstance(source, InitialConditionTimeDataset):
            self.base_data = source.base_data
            self.indices = source.indices
        else:
            base_ds, indices = _unwrap_dataset_and_indices(source)

            if torch.is_tensor(source):
                self.base_data = source
                self.indices = None
            else:
                if not hasattr(base_ds, "data"):
                    raise ValueError(
                        "InitialConditionTimeDataset expects source with a `.data` tensor of shape (N,S,S,T)."
                    )
                self.base_data = base_ds.data
                self.indices = indices

        if self.base_data.dim() != 4:
            raise ValueError(f"Expected trajectory tensor with 4 dims (N,S,S,T); got shape {tuple(self.base_data.shape)}")

        self.time_sampling = str(time_sampling).lower()
        self.fixed_t_idx = int(fixed_t_idx)
        self.min_t_idx = int(min_t_idx)
        self.time_encoding = str(time_encoding).lower()
        self.time_interval = float(time_interval)
        self.max_time_index = int(self.base_data.shape[-1] - 1)
        if self.max_time_index < 0:
            raise ValueError("Trajectory tensor must have at least one time slice.")
        if self.max_time_index < self.min_t_idx:
            raise ValueError(
                f"min_t_idx={self.min_t_idx} exceeds max_time_index={self.max_time_index}; "
                "need at least two time slices to sample targets."
            )

    def __len__(self):
        return len(self.indices) if self.indices is not None else int(self.base_data.shape[0])

    def _encode_time(self, t_idx: int) -> float:
        if self.time_encoding in ("normalized", "norm"):
            return float(t_idx) / float(max(1, self.max_time_index))
        if self.time_encoding in ("physical", "phys"):
            return (float(t_idx) / float(max(1, self.max_time_index))) * self.time_interval
        if self.time_encoding in ("index", "step", "steps"):
            return float(t_idx)
        raise ValueError(f"Unknown time_encoding={self.time_encoding!r}; use normalized|physical|index.")

    def _sample_t_idx(self) -> int:
        if self.time_sampling == "fixed":
            return max(self.min_t_idx, min(self.fixed_t_idx, self.max_time_index))
        if self.time_sampling == "random":
            return int(torch.randint(self.min_t_idx, self.max_time_index + 1, ()).item())
        raise ValueError(f"Unknown time_sampling={self.time_sampling!r}; use random|fixed.")

    def get_trajectory(self, idx):
        base_idx = int(self.indices[idx]) if self.indices is not None else int(idx)
        return self.base_data[base_idx]

    def __getitem__(self, idx):
        traj = self.get_trajectory(idx)
        t_idx = self._sample_t_idx()
        x0 = traj[..., 0]
        y = traj[..., t_idx]
        t = torch.tensor(self._encode_time(t_idx), dtype=torch.float32)
        return x0, t, y


def _build_x0_grid_t_input(x0, grid, t):
    """
    Build model input from (x0, grid, t).

    - x0: (B,S,S)
    - grid: (1,S,S,2)
    - t: (B,) float32
    Returns: (B,S,S,4)
    """
    batch, sx, sy = x0.shape[0], x0.shape[1], x0.shape[2]
    t_chan = t.view(batch, 1, 1, 1).expand(batch, sx, sy, 1)
    return torch.cat((x0.unsqueeze(-1), grid.expand(batch, -1, -1, -1), t_chan), dim=-1)


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
            if isinstance(pred, tuple):
                pred = pred[0]
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


def evaluate_from_initial(model, test_loader, device, grid):
    """Evaluate prediction u_0 -> u(t) conditioned on scalar t."""
    lploss = LpLoss(size_average=True)
    model.eval()
    total = 0.0
    batches = 0
    pred_plot = None
    target_plot = None
    with torch.no_grad():
        for x0, t, y in test_loader:
            x0, t, y = x0.to(device), t.to(device), y.to(device)
            x_in = _build_x0_grid_t_input(x0, grid, t)
            pred = model(x_in)
            if isinstance(pred, tuple):
                pred = pred[0]
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


def _infer_max_time_index(source) -> int:
    if isinstance(source, DataLoader):
        source = source.dataset
    if isinstance(source, InitialConditionTimeDataset):
        return int(source.max_time_index)
    base_ds, _ = _unwrap_dataset_and_indices(source)
    if hasattr(base_ds, "max_time_index"):
        return int(base_ds.max_time_index)
    if hasattr(base_ds, "data"):
        return int(base_ds.data.shape[-1] - 1)
    raise ValueError("Cannot infer max_time_index; expected source with `.data` or `.max_time_index`.")


def evaluate_from_initial_fixed_time(
    model,
    test_source,
    device,
    grid,
    t_idx,
    batch_size,
    time_encoding="normalized",
    time_interval=1.0,
):
    """Evaluate u_0 -> u(t) using a fixed time index across the test set."""
    if isinstance(test_source, DataLoader):
        test_source = test_source.dataset
    ds = InitialConditionTimeDataset(
        test_source,
        time_sampling="fixed",
        fixed_t_idx=int(t_idx),
        min_t_idx=1,
        time_encoding=time_encoding,
        time_interval=time_interval,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate_from_initial(model, loader, device, grid)


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
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.dim() == 5:
            pred = pred.squeeze(-2)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
    return pred, y.unsqueeze(0)


def get_fixed_initial_condition_pair(model, test_source, grid, device, sample_idx=0, t_idx=1, time_encoding="normalized", time_interval=1.0):
    """Grab a deterministic (x0, t) -> x_t example for visualization."""
    source = test_source.dataset if isinstance(test_source, DataLoader) else test_source
    if isinstance(source, InitialConditionTimeDataset):
        max_t = source.max_time_index
        traj = source.get_trajectory(min(sample_idx, len(source) - 1))
    else:
        base_ds, indices = _unwrap_dataset_and_indices(source)
        if not hasattr(base_ds, "data"):
            return None, None
        data = base_ds.data
        max_t = int(data.shape[-1] - 1)
        sample_idx = min(sample_idx, (len(indices) - 1) if indices is not None else (data.shape[0] - 1))
        base_idx = indices[sample_idx] if indices is not None else sample_idx
        traj = data[base_idx]

    if max_t <= 0:
        return None, None
    t_idx = max(1, min(int(t_idx), max_t))
    x0 = traj[..., 0].to(device)
    y = traj[..., t_idx].to(device)

    if str(time_encoding).lower() in ("normalized", "norm"):
        t_val = float(t_idx) / float(max(1, max_t))
    elif str(time_encoding).lower() in ("physical", "phys"):
        t_val = (float(t_idx) / float(max(1, max_t))) * float(time_interval)
    else:
        t_val = float(t_idx)

    t = torch.tensor([t_val], dtype=torch.float32, device=device)
    grid_b = grid.to(device)
    x_in = _build_x0_grid_t_input(x0.unsqueeze(0), grid_b, t)
    with torch.no_grad():
        pred = model(x_in)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.dim() == 5:
            pred = pred.squeeze(-2)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
    return pred, y.unsqueeze(0)


def _parse_epoch_from_name(save_name, fname):
    """Infer epoch from fname given a canonical save_name."""
    stem, ext = os.path.splitext(save_name)
    if fname == save_name:
        return 0
    if not fname.startswith(stem + "_") or not fname.endswith(ext):
        return -1
    suffix = fname[len(stem) + 1:-len(ext)]
    try:
        return int(suffix)
    except ValueError:
        return -1

def train_step_ahead(model, train_loader, optimizer, scheduler, config, device, grid, test_loader=None, eval_step=100,save_step=1000, use_tqdm=True, writer=None, model_name='fno2d', start_ep=0):
    """Train on one-step pairs (u_t, u_{t+1})."""
    lploss = LpLoss(size_average=True)
    epochs = config['train']['epochs']
    grid = grid.to(device).unsqueeze(0)

    lambda_amp_final = 1e-2    # good starting point
    warmup_frac = 0.2          # first 20% epochs
    if start_ep >= epochs:
        print(f'start_ep ({start_ep}) >= epochs ({epochs}); skipping training loop.')
        return
    if use_tqdm:
        pbar = tqdm(range(start_ep, epochs), dynamic_ncols=True, smoothing=0.1)
    else:
        pbar = range(start_ep, epochs)
    for ep in pbar:
        model.train()
        running = 0.0
        batches = 0

        # linear warm-up
        t = ep / max(1, epochs - 1)
        if t < warmup_frac:
            lambda_amp = 0.0
        else:
            ramp = (t - warmup_frac) / (1.0 - warmup_frac)
            lambda_amp = lambda_amp_final * ramp
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch = x.shape[0]
            x_in = torch.cat((x.unsqueeze(-1), grid.expand(batch, -1, -1, -1)), dim=-1)
            pred = model(x_in)
            if isinstance(pred, tuple):
                pred, x_reg = pred
                # print("pred shape:", pred.shape, "x_reg shape:", x_reg.shape)
            else:
                x_reg = None
            if pred.dim() == 5:
                pred = pred.squeeze(-2)
            if pred.dim() == 4:
                pred = pred.squeeze(-1)
        

            data_loss = lploss(pred, y)
            if x_reg is not None:
                loss = data_loss + lambda_amp * x_reg
            else:
                loss = data_loss

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


def train_from_initial_time(
    model,
    train_loader,
    optimizer,
    scheduler,
    config,
    device,
    grid,
    test_loader=None,
    warmup_loader=None,
    warmup_steps=0,
    eval_step=100,
    save_step=1000,
    use_tqdm=True,
    writer=None,
    model_name="fno2d",
    start_ep=0,
):
    """
    Train on (x0, t) -> x(t) with optional warmup on fixed t=1 for N optimizer steps.

    Expected loader batches:
      - warmup_loader: (x0, t, y) with fixed t
      - train_loader:  (x0, t, y) with sampled t
    """
    lploss = LpLoss(size_average=True)
    train_cfg = config["train"]
    data_cfg = config.get("data", {})
    epochs = train_cfg["epochs"]
    grid = grid.to(device).unsqueeze(0)

    lambda_amp_final = 1e-2
    warmup_frac = 0.2

    if warmup_steps > 0 and warmup_loader is not None and start_ep == 0:
        model.train()
        it = iter(warmup_loader)
        running = 0.0
        for step in range(int(warmup_steps)):
            try:
                x0, t, y = next(it)
            except StopIteration:
                it = iter(warmup_loader)
                x0, t, y = next(it)
            x0, t, y = x0.to(device), t.to(device), y.to(device)
            x_in = _build_x0_grid_t_input(x0, grid, t)
            pred = model(x_in)
            if isinstance(pred, tuple):
                pred, x_reg = pred
            else:
                x_reg = None
            if pred.dim() == 5:
                pred = pred.squeeze(-2)
            if pred.dim() == 4:
                pred = pred.squeeze(-1)
            data_loss = lploss(pred, y)
            loss = data_loss if x_reg is None else (data_loss + 0.0 * x_reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (step + 1) % max(1, min(100, warmup_steps // 10)) == 0:
                avg = running / float(step + 1)
                print(f"Warmup step {step + 1}/{warmup_steps}, train L2: {avg:.6f}")
                if writer is not None:
                    writer.add_scalar("train/warmup_l2", avg, step + 1)

    if start_ep >= epochs:
        print(f"start_ep ({start_ep}) >= epochs ({epochs}); skipping training loop.")
        return

    if use_tqdm:
        pbar = tqdm(range(start_ep, epochs), dynamic_ncols=True, smoothing=0.1)
    else:
        pbar = range(start_ep, epochs)

    time_encoding = train_cfg.get("time_encoding", "normalized")
    time_interval = data_cfg.get("time_interval", 1.0)
    fixed_eval_t_idx = int(train_cfg.get("fixed_eval_t_idx", 1))
    eval_batchsize = int(train_cfg.get("eval_batchsize", train_cfg.get("batchsize", 1)))
    save_eval_preds = bool(train_cfg.get("save_eval_predictions", True))
    eval_pred_dir = os.path.join(train_cfg["save_dir"], "eval_predictions")
    if save_eval_preds:
        os.makedirs(eval_pred_dir, exist_ok=True)

    for ep in pbar:
        model.train()
        running = 0.0
        batches = 0

        t_epoch = ep / max(1, epochs - 1)
        if t_epoch < warmup_frac:
            lambda_amp = 0.0
        else:
            ramp = (t_epoch - warmup_frac) / (1.0 - warmup_frac)
            lambda_amp = lambda_amp_final * ramp

        for x0, t, y in train_loader:
            x0, t, y = x0.to(device), t.to(device), y.to(device)
            x_in = _build_x0_grid_t_input(x0, grid, t)
            pred = model(x_in)
            if isinstance(pred, tuple):
                pred, x_reg = pred
            else:
                x_reg = None
            if pred.dim() == 5:
                pred = pred.squeeze(-2)
            if pred.dim() == 4:
                pred = pred.squeeze(-1)

            data_loss = lploss(pred, y)
            if x_reg is not None:
                loss = data_loss + lambda_amp * x_reg
            else:
                loss = data_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1

        scheduler.step()
        avg = running / max(1, batches)
        print(f"Epoch {ep + 1}/{epochs}, train L2: {avg:.6f}")
        if writer is not None:
            writer.add_scalar("train/l2", avg, ep + 1)
        if use_tqdm:
            pbar.set_description((f"Train L2: {avg:.6f}"))

        if ep % eval_step == 0 and test_loader is not None:
            max_t_idx = max(1, _infer_max_time_index(test_loader))
            test_l2_1, _, _ = evaluate_from_initial_fixed_time(
                model,
                test_loader,
                device,
                grid,
                t_idx=1,
                batch_size=eval_batchsize,
                time_encoding=time_encoding,
                time_interval=time_interval,
            )
            test_l2_max, _, _ = evaluate_from_initial_fixed_time(
                model,
                test_loader,
                device,
                grid,
                t_idx=max_t_idx,
                batch_size=eval_batchsize,
                time_encoding=time_encoding,
                time_interval=time_interval,
            )
            print(f"Test relative L2 (t=1): {test_l2_1:.6f} | (t=max={max_t_idx}): {test_l2_max:.6f}")
            if writer is not None:
                writer.add_scalar("eval/test_l2_t1", test_l2_1, ep + 1)
                writer.add_scalar("eval/test_l2_tmax", test_l2_max, ep + 1)

                fixed_pred_t1, fixed_target_t1 = get_fixed_initial_condition_pair(
                    model,
                    test_loader,
                    grid,
                    device,
                    sample_idx=0,
                    t_idx=1,
                    time_encoding=time_encoding,
                    time_interval=time_interval,
                )
                if fixed_pred_t1 is not None:
                    log_tensorboard_images_and_spectra(
                        writer,
                        fixed_pred_t1.unsqueeze(-1),
                        fixed_target_t1.unsqueeze(-1),
                        ep + 1,
                        "vorticity_t1",
                        model_name,
                    )

                fixed_pred_tmax, fixed_target_tmax = get_fixed_initial_condition_pair(
                    model,
                    test_loader,
                    grid,
                    device,
                    sample_idx=0,
                    t_idx=max_t_idx,
                    time_encoding=time_encoding,
                    time_interval=time_interval,
                )
                if fixed_pred_tmax is not None:
                    log_tensorboard_images_and_spectra(
                        writer,
                        fixed_pred_tmax.unsqueeze(-1),
                        fixed_target_tmax.unsqueeze(-1),
                        ep + 1,
                        "vorticity_tmax",
                        model_name,
                    )
                    if save_eval_preds:
                        torch.save(
                            {
                                "epoch": int(ep + 1),
                                "t_idx": int(max_t_idx),
                                "pred": fixed_pred_tmax.detach().cpu(),
                                "target": fixed_target_tmax.detach().cpu(),
                                "test_l2_t1": float(test_l2_1),
                                "test_l2_tmax": float(test_l2_max),
                            },
                            os.path.join(eval_pred_dir, f"tmax_pred_ep{ep + 1}.pt"),
                        )

        if ep % save_step == 0 and ep > 0:
            save_checkpoint(
                train_cfg["save_dir"],
                train_cfg["save_name"].replace(".pt", f"_{ep + 1}.pt"),
                model,
                optimizer,
                scheduler,
            )


def build_synthetic_dataset(data_config, n_samples, step_ahead=False, time_conditioned=False):
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

    if time_conditioned:
        data = torch.rand(n_samples, S, S, T)

        class SyntheticTrajectoryDataset(Dataset):
            def __init__(self, arr):
                self.data = arr

            def __len__(self):
                return self.data.shape[0]

            def __getitem__(self, idx):
                return self.data[idx]

        return SyntheticTrajectoryDataset(data), S, T

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
    train_cfg = config.get('train', {})
    train_mode = str(train_cfg.get('mode', 'step_ahead')).lower()
    use_time_conditioned = train_mode in {
        'from_initial',
        'from_initial_time',
        'time_conditioned',
        'time_dependent',
        'multistep',
        'multi_step',
    }

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, T_data = build_synthetic_dataset(
            data_config,
            args.synthetic_samples,
            step_ahead=not use_time_conditioned,
            time_conditioned=use_time_conditioned,
        )
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
        if use_time_conditioned:
            time_encoding = train_cfg.get("time_encoding", "normalized")
            eval_time_sampling = train_cfg.get("eval_time_sampling", "random")
            fixed_eval_t_idx = train_cfg.get("fixed_eval_t_idx", 1)
            test_ds = InitialConditionTimeDataset(
                test_set,
                time_sampling=eval_time_sampling,
                fixed_t_idx=fixed_eval_t_idx,
                min_t_idx=train_cfg.get("min_t_idx", 1),
                time_encoding=time_encoding,
                time_interval=data_config.get("time_interval", 1.0),
            )
            test_loader = DataLoader(test_ds, batch_size=train_cfg["batchsize"], shuffle=False)
        else:
            test_loader = DataLoader(test_set, batch_size=train_cfg["batchsize"], shuffle=False)
    else:
        train_set = full_dataset
        test_loader = None

    warmup_loader = None
    if use_time_conditioned:
        time_encoding = train_cfg.get("time_encoding", "normalized")
        train_ds = InitialConditionTimeDataset(
            train_set,
            time_sampling="random",
            fixed_t_idx=train_cfg.get("fixed_eval_t_idx", 1),
            min_t_idx=train_cfg.get("min_t_idx", 1),
            time_encoding=time_encoding,
            time_interval=data_config.get("time_interval", 1.0),
        )
        train_loader = DataLoader(train_ds, batch_size=train_cfg["batchsize"], shuffle=data_config["shuffle"])
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
        if warmup_steps > 0:
            warmup_ds = InitialConditionTimeDataset(
                train_set,
                time_sampling="fixed",
                fixed_t_idx=int(train_cfg.get("warmup_time_idx", 1)),
                min_t_idx=1,
                time_encoding=time_encoding,
                time_interval=data_config.get("time_interval", 1.0),
            )
            warmup_loader = DataLoader(warmup_ds, batch_size=train_cfg["batchsize"], shuffle=True)
    else:
        train_loader = DataLoader(train_set, batch_size=train_cfg["batchsize"], shuffle=data_config["shuffle"])
        warmup_steps = 0

    # create model
    print("device: ", device)
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'fno2d').lower()
    default_in_dim = 4 if use_time_conditioned else 3
    in_dim = model_cfg.get("in_dim", model_cfg.get("in_chans", default_in_dim))
    
    if model_name == 'hfs':
        model = ResUNet(in_c=in_dim,
                        out_c=1,
                        target_params=model_cfg.get('target_params', 'medium'),
                        device=device).to(device)
    elif model_name in ['wno', 'wno2d']:
        dummy = torch.zeros(1, 1, S_data, S_data, device=device)
        model = WNO2d(in_channels=model_cfg.get('in_chans', in_dim),
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
            in_chans=model_cfg.get('in_chans', in_dim),
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
            input_dim=model_cfg.get('in_chans', in_dim),  # expecting u with grid/time concatenated
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', 128),
            n_layers=model_cfg.get('n_layers', 5),
            patch_size= model_cfg.get('patch_size', None),
        ).to(device)
    elif model_name in ['multiscale_wavelet', 'multiscale_wavelet2d', 'multiscale_wavelet_transformer2d']:
        model = MultiscaleWaveletTransformer2D(
            wave=model_cfg.get('wave', 'haar'),
            input_dim=model_cfg.get('in_chans', in_dim),
            output_dim=model_cfg.get('out_chans', 1),
            dim=model_cfg.get('dim', 128),
            n_layers=model_cfg.get('n_layers', 5),
            patch_size= model_cfg.get('patch_size', None),
            use_efficient_attention=model_cfg.get('use_efficient_attention', False),
            efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2]),
        ).to(device)
    elif model_name in ['mswt_stable2d']:
        model = MSWT2DStable(input_dim=model_cfg.get('in_chans', in_dim),
                             output_dim=model_cfg.get('out_chans', 1),
                             dim=model_cfg.get('dim', 128),
                             n_layers=model_cfg.get('n_layers', 5),
                             use_efficient_attention=model_cfg.get('use_efficient_attention', True),
                             efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2])).to(device)
    elif model_name in ['mswt_stable_soft_control2d']:
        model = MSWT2DStableSoftControl(input_dim=model_cfg.get('in_chans', in_dim),
                             output_dim=model_cfg.get('out_chans', 1),
                             dim=model_cfg.get('dim', 128),
                             n_layers=model_cfg.get('n_layers', 5),
                             use_efficient_attention=model_cfg.get('use_efficient_attention', True),
                             efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2])).to(device)
    elif model_name in ['mswt_stable_norm_energy2d']:
        model = MSWT2DStableNormalizedEnergy(input_dim=model_cfg.get('in_chans', in_dim),
                             output_dim=model_cfg.get('out_chans', 1),
                             dim=model_cfg.get('dim', 128),
                             n_layers=model_cfg.get('n_layers', 5),
                             use_efficient_attention=model_cfg.get('use_efficient_attention', True),
                             efficient_layers=model_cfg.get('efficient_layers', [0, 1, 2])).to(device)
    elif model_name in ['saot', 'saot2d']:
        default_fun_dim = 1 + (1 if use_time_conditioned else 0)
        model = SAOTModel(space_dim=model_cfg.get('space_dim', 2),
                        n_layers=model_cfg.get('n_layers', 3),
                        n_hidden=model_cfg.get('n_hidden', 64)  ,
                        dropout=model_cfg.get('dropout', 0.0),
                        n_head=model_cfg.get('n_head', 4),
                        Time_Input=model_cfg.get('Time_Input', False),
                        mlp_ratio=model_cfg.get('mlp_ratio', 1),
                        fun_dim=model_cfg.get('fun_dim', default_fun_dim),
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
                      in_dim=model_cfg.get('in_dim', in_dim),
                      out_dim=model_cfg.get('out_dim', 1),
                      act=model_cfg['act'],
                    #   pad_ratio=model_cfg.get('pad_ratio', [0., 0.])
                      ).to(device)
    print('model structure: ', model)
    
    
    # create optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])

    start_ep = 0
    if args.resume_training:
        ckpt_path = os.path.join(config['train']['save_dir'], args.resume_ckpt)
        parsed_ep = _parse_epoch_from_name(config['train']['save_name'], args.resume_ckpt)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            if ckpt.get('optim') is not None:
                optimizer.load_state_dict(ckpt['optim'])
            if ckpt.get('scheduler') is not None:
                scheduler.load_state_dict(ckpt['scheduler'])
                sched_epoch = scheduler.state_dict().get('last_epoch', -1) + 1
                parsed_ep = max(parsed_ep, sched_epoch)
            start_ep = max(parsed_ep, 0)
            print(f'Weights loaded from {ckpt_path}, resuming at epoch {start_ep + 1}')
        else:
            print('resume_training requested but no checkpoint found; starting from scratch.')
    save_dir = config['train']['save_dir'] if torch.cuda.is_available() else 'saved_models'
    tensorboard_dir = config['train'].get('tensorboard_dir')
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    grid = torch2dgrid(S_data, S_data)
    if use_time_conditioned:
        train_from_initial_time(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            device,
            grid,
            test_loader=test_loader,
            warmup_loader=warmup_loader,
            warmup_steps=warmup_steps,
            writer=writer,
            model_name=model_name,
            start_ep=start_ep,
        )
    else:
        train_step_ahead(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            device,
            grid,
            test_loader=test_loader,
            writer=writer,
            model_name=model_name,
            start_ep=start_ep,
        )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer, scheduler)
    if test_loader is not None:
        if use_time_conditioned:
            max_t_idx = max(1, _infer_max_time_index(test_loader))
            test_l2_1, _, _ = evaluate_from_initial_fixed_time(
                model,
                test_loader,
                device,
                grid.to(device).unsqueeze(0),
                t_idx=1,
                batch_size=int(train_cfg.get("eval_batchsize", train_cfg.get("batchsize", 1))),
                time_encoding=train_cfg.get("time_encoding", "normalized"),
                time_interval=data_config.get("time_interval", 1.0),
            )
            test_l2_max, _, _ = evaluate_from_initial_fixed_time(
                model,
                test_loader,
                device,
                grid.to(device).unsqueeze(0),
                t_idx=max_t_idx,
                batch_size=int(train_cfg.get("eval_batchsize", train_cfg.get("batchsize", 1))),
                time_encoding=train_cfg.get("time_encoding", "normalized"),
                time_interval=data_config.get("time_interval", 1.0),
            )
            if bool(train_cfg.get("save_eval_predictions", True)):
                eval_pred_dir = os.path.join(train_cfg["save_dir"], "eval_predictions")
                os.makedirs(eval_pred_dir, exist_ok=True)
                fixed_pred_tmax, fixed_target_tmax = get_fixed_initial_condition_pair(
                    model,
                    test_loader,
                    grid.to(device).unsqueeze(0),
                    device,
                    sample_idx=0,
                    t_idx=max_t_idx,
                    time_encoding=train_cfg.get("time_encoding", "normalized"),
                    time_interval=data_config.get("time_interval", 1.0),
                )
                if fixed_pred_tmax is not None:
                    torch.save(
                        {
                            "epoch": int(config["train"]["epochs"]),
                            "t_idx": int(max_t_idx),
                            "pred": fixed_pred_tmax.detach().cpu(),
                            "target": fixed_target_tmax.detach().cpu(),
                            "test_l2_t1": float(test_l2_1),
                            "test_l2_tmax": float(test_l2_max),
                        },
                        os.path.join(eval_pred_dir, "tmax_pred_final.pt"),
                    )
        else:
            test_l2, _, _ = evaluate_step_ahead(model, test_loader, device, grid.to(device).unsqueeze(0))
        if writer is not None:
            if use_time_conditioned:
                writer.add_scalar("eval/test_l2_t1", test_l2_1, config["train"]["epochs"])
                writer.add_scalar("eval/test_l2_tmax", test_l2_max, config["train"]["epochs"])
            else:
                writer.add_scalar('eval/test_l2', test_l2, config['train']['epochs'])
        if use_time_conditioned:
            print(f"Test relative L2 (t=1): {test_l2_1:.6f} | (t=max={max_t_idx}): {test_l2_max:.6f}")
        else:
            print(f'Random test split relative L2: {test_l2:.6f}')
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
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Specific checkpoint filename to resume from (in save_dir)')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    train_3d(args, config)

# Training an FNO with `train_operator.py` on `NS3DDataset`

This note walks through the minimal steps to launch Fourier Neural Operator (FNO) training on Navier–Stokes data that follow the `NS3DDataset` format, using the helper script `train_operator.py`.

## What the script expects
- Input data: one or two `.npy` files shaped `(num_samples, Nt, Nx, Ny)`; this is the same layout produced/consumed by `train_utils.datasets.NS3DDataset`. `datapath`/`datapath2` point to these files.
- Resolution controls: `nx`/`nt` are the raw spatial/time sizes; `sub` and `sub_t` downsample them; `time_interval` lets you train on 0.5 s windows via the built‑in `extract` logic.
- Loader split: `n_sample` and `offset` pick the training slice; anything after that slice is left unused by this script.
- Mixed training: `data_iter` and `eqn_iter` set how many supervised vs. equation-only steps run per epoch; `ic_loss`, `f_loss`, and `xy_loss` weight the corresponding losses.
- Model: `modes1/2/3`, `layers`, `fc_dim`, and `act` parameterize `FNO3d`. `pad_ratio` is optional and defaults to no padding.
- Logging/checkpointing: `save_dir` + `save_name` decide where the final checkpoint lands; add `train.ckpt` to resume. Pass `--log` and supply `log.entity/project/group` to enable Weights & Biases.

## Example config snippet
```yaml
data:
  datapath: 'data/NS_fine_Re500_T128_part0.npy'  # shape: (N, 129, 128, 128)
  Re: 500
  total_num: 100
  offset: 0
  n_sample: 80
  time_interval: 1.0          # use 0.5 to enable overlapping windows
  nx: 128
  nt: 128
  sub: 2
  sub_t: 1
  shuffle: true
  S2: 64                      # spatial grid for equation-only steps
  T2: 65                      # time grid for equation-only steps

model:
  layers: [64, 64, 64, 64, 64]
  modes1: [8, 8, 8, 8]
  modes2: [8, 8, 8, 8]
  modes3: [8, 8, 8, 8]
  fc_dim: 128
  act: gelu

train:
  batchsize: 2
  epochs: 400
  milestones: [200, 300]
  base_lr: 0.001
  scheduler_gamma: 0.5
  ic_loss: 0.0                # set >0 to include initial-condition loss
  f_loss: 0.0                 # set >0 to include PDE residual loss
  xy_loss: 1.0
  data_iter: 1                # number of supervised steps per epoch
  eqn_iter: 0                 # set >0 to mix in equation-only steps
  save_dir: 'checkpoints/re500'
  save_name: 'fno-re500.pt'

log:
  project: 'PINO-NS'
  group: 'Re500-FNO'
  entity: 'your-wandb-handle' # only needed if you run with --log
```

## Run training
```bash
python3 train_operator.py --config_path configs/pretrain/Re500-pretrain-1s.yaml
# add --log if your config has log.entity/project/group and wandb is installed
```
The script auto-selects GPU if available. It builds `NSLoader` instances that wrap the same `(N, Nt, Nx, Ny)` arrays as `NS3DDataset` and trains `FNO3d`. To resume, set `train.ckpt` in the config to a checkpoint created earlier.

## Outputs and follow-ups
- The final checkpoint is written to `train.save_dir/train.save_name`; intermediate checkpoints are not emitted by default.
- Use `eval_operator.py` with a matching test config to measure error on held-out trajectories.
- If you mix in PDE residuals (`ic_loss`/`f_loss` > 0), ensure `S2`/`T2` match the grid you want for the physics loss; otherwise set `eqn_iter` to 0 for data-only FNO training.

import argparse
import os
import torch
import numpy as np

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_trajectories_path', type=str, default='../../results/trajectories/sgan')
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def flatten_scene(trajs_arr, frame_nums=None, ped_nums=None, frame_skip=10):
    """flattens a 3d scene of shape (num_peds, ts, 2) into a 2d array of shape (num_peds, ts x 4)
    ped_nums (optional): list of ped numbers to assign, length == num_peds
    frame_nums (optional): list of frame numbers to assign to the resulting, length == ts
    """
    assert len(trajs_arr.shape) == 3
    if ped_nums is None:
        ped_nums = np.arange(0, trajs_arr.shape[0])
    if frame_nums is None:
        frame_nums = np.arange(0, trajs_arr.shape[1] * frame_skip, frame_skip)
    # extend num peds by num of timesteps
    ped_ids = np.tile(np.array(ped_nums).reshape(-1, 1), (1, trajs_arr.shape[1])).reshape(-1, 1)
    # extend num frames by num of timesteps
    frame_ids = np.tile(np.array(frame_nums).reshape(1, -1), (trajs_arr.shape[0], 1)).reshape(-1, 1)
    assert ped_ids.shape == frame_ids.shape, f"ped_ids.shape: {ped_ids.shape}, != frame_ids.shape: {frame_ids.shape}"
    trajs_arr = np.concatenate([frame_ids, ped_ids, trajs_arr.reshape(-1, 2)], -1)
    assert trajs_arr.shape == (len(ped_nums) * len(frame_nums) , 4), \
        f"trajs_arr.shape: {trajs_arr.shape}, != {(ped_nums * frame_nums , 4)}"
    return trajs_arr


# sanity checks for same dataset size as the standard
dset_to_num_peds = {
        'eth': 364,
        'hotel': 1197,  # but has 1075
        'univ': 14295 + 10039,
        'zara1': 2356,
        'zara2': 5910,
        'sdd': 2829,
}
dset_to_num_frames = {
        'eth': 253,
        'hotel': 445,  # but has 1075
        'univ': 425 + 522,
        'zara1': 705,
        'zara2': 998,
        'sdd': 1999,
}

SEQUENCE_NAMES = {
        'eth': ['biwi_eth'],
        'hotel': ['biwi_hotel'],
        'zara1': ['crowds_zara01'],
        'zara2': ['crowds_zara02'],
        'univ': ['students001', 'students003'],
        'sdd': [
                'coupa_0', 'coupa_1', 'gates_2', 'hyang_0', 'hyang_1', 'hyang_3',
                'hyang_8', 'little_0', 'little_1', 'little_2', 'little_3',
                'nexus_5', 'nexus_6', 'quad_0', 'quad_1', 'quad_2', 'quad_3',
        ],
}


def format_and_save_trajectories(save_dir, seq_names, frame_ids, pred_traj_fakes, pred_traj_gt, obs_traj, seq_start_end):
    pred_traj_gt = pred_traj_gt.cpu().numpy()
    obs_traj = obs_traj.cpu().numpy()
    obs_len = 8
    pred_len = 12

    for i, (s, e) in enumerate(seq_start_end):
        frame_id = frame_ids[i][0]
        seq_name = seq_names[i][0]

        ot = obs_traj[:, s:e].swapaxes(0, 1)
        obs_traj_f = flatten_scene(ot, frame_nums=np.arange(frame_id, frame_id+obs_len))
        save_trajectories(obs_traj_f, save_dir, seq_name, frame_id, suffix="/obs")

        pred_traj_gt_f = flatten_scene(pred_traj_gt[:, s:e].swapaxes(0,1), frame_nums=np.arange(
                frame_id+obs_len, frame_id+obs_len+pred_len))
        save_trajectories(pred_traj_gt_f, save_dir, seq_name, frame_id, suffix=f"/gt")

        for ptf_i, pred_traj_fake in enumerate(pred_traj_fakes):
            ptf = pred_traj_fake[:, s:e].cpu().numpy().swapaxes(0,1)
            pred_traj_fake_f = flatten_scene(ptf, frame_nums=np.arange(frame_id+obs_len, frame_id + obs_len+pred_len))
            save_trajectories(pred_traj_fake_f, save_dir, seq_name, frame_id, suffix=f"/sample_{ptf_i:03d}")


def save_trajectories(trajectory, save_dir, seq_name, frame, suffix=''):
    """Save trajectories in a text file.
    Input:
        trajectory: (np.array/torch.Tensor) Predcited trajectories with shape
                    of (n_pedestrian, future_timesteps, 4). The last elemen is
                    [frame_id, track_id, x, y] where each element is float.
        save_dir: (str) Directory to save into.
        seq_name: (str) Sequence name (e.g., eth_biwi, coupa_0)
        frame: (num) Frame ID.
        suffix: (str) Additional suffix to put into file name.
    """
    fname = f"{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt"
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    np.savetxt(fname, trajectory, fmt="%.3f")


def evaluate(args, loader, generator, num_samples, trajectories_save_path):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() if isinstance(tensor, torch.Tensor) else tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, seq_name, frame_id) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            pred_traj_fakes = []
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
                pred_traj_fakes.append(pred_traj_fake)

            format_and_save_trajectories(trajectories_save_path, seq_name, frame_id, pred_traj_fakes, pred_traj_gt, obs_traj, seq_start_end)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        if '12' not in path:  # only evaluate pred_len=12
            continue
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples, args.save_trajectories_path)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


#Retrieved from https://github.com/sshaoshuai/MTR, licensed under the Apache 2.0 license.

#Modified by Trent Weiss starting 9/28/2023

import numpy as np
import torch
import scipy.interpolate
from scipy.spatial.transform import Rotation, RotationSpline

def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, :cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor

def collate_batch(batch_list):
    """
    Args:
    batch_list:
        scenario_id: (num_center_objects)
        track_index_to_predict (num_center_objects):

        obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
        map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

        obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
        obj_trajs_last_pos: (num_center_objects, num_objects, 3)
        obj_types: (num_objects)
        obj_ids: (num_objects)

        center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        center_objects_type: (num_center_objects)
        center_objects_id: (num_center_objects)

        obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
        obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
        center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
        center_gt_trajs_mask (num_center_objects, num_future_timestamps):
        center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
    """
    batch_size = len(batch_list)
    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

    input_dict = {}
    for key, val_list in key_to_list.items():

        if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
            'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
            val_list = [torch.from_numpy(x) for x in val_list]
            input_dict[key] = merge_batch_by_padding_2nd_dim(val_list)
        elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
            input_dict[key] = np.concatenate(val_list, axis=0)
        else:
            val_list = [torch.from_numpy(x) for x in val_list]
            input_dict[key] = torch.cat(val_list, dim=0)

    batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]
    batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
    return batch_dict


def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask

def create_map_data_for_center_objects(all_polylines : np.ndarray, dataset_cfg : dict):
        """
        Args:
            all_polylines (num_center_objects, num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type] Assumed to already be in local coordinates
            
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = all_polylines.shape[0]


        map_polylines_list = []
        map_polylines_mask_list = []
        for j in range(num_center_objects):
            batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
                polylines=all_polylines[j], point_sampled_interval=dataset_cfg['POINT_SAMPLED_INTERVAL'],
                vector_break_dist_thresh=dataset_cfg['VECTOR_BREAK_DIST_THRESH'],
                num_points_each_polyline=dataset_cfg['NUM_POINTS_EACH_POLYLINE'],
            )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)
            map_polylines_list.append(batch_polylines)
            map_polylines_mask_list.append(batch_polylines_mask)

        # collect a number of closest polylines for each center objects

        map_polylines = torch.stack(map_polylines_list, dim=0)
        map_polylines_mask = torch.stack(map_polylines_mask_list, dim=0)

        xy_pos_pre = map_polylines[:, :, :, 0:2]
        xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        map_polylines = torch.cat((map_polylines, xy_pos_pre), dim=-1)
        map_polylines[map_polylines_mask==0]=0.0

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

        map_polylines = map_polylines.numpy()
        map_polylines_mask = map_polylines_mask.numpy()
        map_polylines_center = map_polylines_center.numpy()

        return map_polylines, map_polylines_mask, map_polylines_center

def generate_centered_trajs_for_agents(obj_trajs_past, obj_types, sdc_index, timestamps, obj_trajs_future):
    """[summary]

    Args:
        obj_trajs_past (num_center_objects, num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_types (num_objects):
        timestamps ([type]): [description]
        obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    Returns:
        ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
        ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
        ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
    """
    assert obj_trajs_past.shape[-1] == 10
    # assert center_objects.shape[-1] == 10
    # num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, box_dim = obj_trajs_past.shape
    num_center_objects = num_objects
    # transform to cpu torch tensor
    # center_objects = torch.from_numpy(center_objects).float()
    obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
    timestamps = torch.from_numpy(timestamps).to(dtype=obj_trajs_past.dtype)

    #There will only be one agent, and will already by in local coordinates
    obj_trajs = obj_trajs_past.unsqueeze(1)


    ## generate the attributes for each object
    object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
    object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
    object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
    object_onehot_mask[torch.arange(num_center_objects), 0, :, 3] = 1
    object_onehot_mask[:, sdc_index, :, 4] = 1

    object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

    object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = torch.cat((
        obj_trajs[:, :, :, 0:6], 
        object_onehot_mask,
        object_time_embedding, 
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9], 
        acce,
    ), dim=-1)

    ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    ##  generate label for future trajectories
    obj_trajs_future = torch.from_numpy(obj_trajs_future).float().unsqueeze(1)

    ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
    ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

    return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()
def create_agent_data_for_center_objects(
        obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
        obj_types, obj_ids
    ):
    #center_objects=center_objects, center_indices=track_index_to_predict,
    obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = generate_centered_trajs_for_agents(
        obj_trajs_past,
        obj_types, 
        sdc_track_index, 
        timestamps, 
        obj_trajs_future
    )

    # generate the labels of track_objects for training
    center_obj_idxs = np.arange(len(track_index_to_predict))
    center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 4)
    center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
    center_gt_trajs[center_gt_trajs_mask == 0] = 0

    # filter invalid past trajs
    assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
    valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

    obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
    obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
    obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
    obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
    obj_types = obj_types[valid_past_mask]
    obj_ids = obj_ids[valid_past_mask]

    valid_index_cnt = valid_past_mask.cumsum(axis=0)
    track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
    sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

    assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
    assert len(obj_types) == obj_trajs_future_mask.shape[1]
    assert len(obj_ids) == obj_trajs_future_mask.shape[1]

    # generate the final valid position of each object
    obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
    obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
    for k in range(num_timestamps):
        cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
        obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

    center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
    for k in range(center_gt_trajs_mask.shape[1]):
        cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
        center_gt_final_valid_idx[cur_valid_mask] = k

    return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
        obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
        track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)

def deepracing_to_mtr(drsample : dict[str, np.ndarray | torch.Tensor], scene_id : str, polyline_config : dict):

    current_position : np.ndarray | torch.Tensor = drsample["current_position"]   
    
    yawonlymask = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=current_position.dtype)

    current_quaternion : np.ndarray = drsample["current_orientation"]
    current_rotation : Rotation = Rotation.from_quat(current_quaternion)
    current_rotation_yawonly : Rotation = Rotation.from_quat(yawonlymask*current_quaternion)
    current_rotmat : np.ndarray = current_rotation.as_matrix()

    thistory : np.ndarray = drsample["thistory"]
    history_points : np.ndarray = drsample["hist"]
    history_vels : np.ndarray = drsample["hist_vel"]
    history_quaternions_full : np.ndarray = drsample["hist_quats"]
    history_quaternions : np.ndarray = history_quaternions_full.copy()
    history_quaternions[:,[0,1]] = 0.0

    tfuture : np.ndarray = drsample["tfuture"][1:]
    future_points : np.ndarray = drsample["fut"][1:]
    future_vels : np.ndarray = drsample["fut_vel"][1:]
    future_quaternions_full : np.ndarray = drsample["fut_quats"][1:]
    future_quaternions : np.ndarray = future_quaternions_full.copy()
    future_quaternions[:,[0,1]] = 0.0

    all_timestamps : np.ndarray = np.concatenate([thistory, tfuture], axis=0)
    all_points = np.concatenate([history_points, future_points], axis=0)
    all_quats = np.concatenate([history_quaternions, future_quaternions], axis=0)
    all_rots : Rotation = Rotation.from_quat(all_quats)
    all_vels = np.concatenate([history_vels, future_vels], axis=0)
    Vo = all_vels[0]
    Vf = all_vels[-1]

    bc_type=([(1, Vo)], [(1, Vf)])
    current_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(all_timestamps, all_points, k=3, bc_type=bc_type)
    current_vel_spline : scipy.interpolate.BSpline = scipy.interpolate.Akima1DInterpolator(all_timestamps, all_vels)
    current_rot_spline : RotationSpline = RotationSpline(all_timestamps, all_rots)

    tnow : float = float(thistory[-1])
    tsamp_past : np.ndarray = np.linspace(-1.0, 0.0, num=11)
    tsamp_future : np.ndarray = np.linspace(0.0, 3.0, num=81)[1:]
    tsamp_waymo : np.ndarray = np.concatenate([tsamp_past, tsamp_future], axis=0) + tnow

    points_waymo = current_spline(tsamp_waymo)
    vels_waymo = current_vel_spline(tsamp_waymo)
    rots_waymo : Rotation = current_rot_spline(tsamp_waymo)
    headings_waymo = rots_waymo.as_rotvec()[:,2]

    # timestamps-=timestamps[-1]
    lb_points : np.ndarray = drsample["left_bd"]
    rb_points : np.ndarray = drsample["right_bd"]

    lb_tangents : np.ndarray = drsample["left_bd_tangents"]
    rb_tangents : np.ndarray = drsample["right_bd_tangents"]


    num_objects = 1
    num_timestamps = tsamp_past.shape[0]
    num_future_timestamps = tsamp_future.shape[0]
    numstates = 10

    dx = 5.63
    dy = 2.0
    dz = 0.95
    dvec = np.asarray([dx, dy, dz], dtype=current_position.dtype)

    #[cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    ego_state_world = np.ones((num_objects, numstates), dtype=current_position.dtype)
    ego_state_world[0, 0:3] = current_position
    ego_state_world[0, 3:6] = dvec
    ego_state_world[0, 6] = current_rotation_yawonly.as_rotvec()[-1]
    ego_state_world[0, 7:9] = (current_rotmat @ history_vels[-1])[:2]

    obj_types : np.ndarray = np.asarray(["TYPE_VEHICLE"])
    obj_trajs_past : np.ndarray = np.ones((num_objects, num_timestamps, numstates), dtype=current_position.dtype)
    obj_trajs_past[0, :, 0:3] = points_waymo[:num_timestamps]
    obj_trajs_past[0, :, 3:6] = np.tile(dvec, num_timestamps).reshape([num_timestamps,3])
    obj_trajs_past[0, :, 6] = headings_waymo[:num_timestamps]
    obj_trajs_past[0, :, 7:9] = vels_waymo[:num_timestamps,0:2]

    obj_trajs_future : np.ndarray = np.ones((num_objects, num_future_timestamps, numstates), dtype=current_position.dtype)
    obj_trajs_future[0, :, 0:3] = points_waymo[num_timestamps:]
    obj_trajs_future[0, :, 3:6] = np.tile(dvec, num_future_timestamps).reshape([num_future_timestamps,3])
    obj_trajs_future[0, :, 6] = headings_waymo[num_timestamps:]
    obj_trajs_future[0, :, 7:9] = vels_waymo[num_timestamps:,0:2]


    sdc_index = sdc_track_index = 0
    track_index_to_predict : np.ndarray = np.asarray([0,], dtype=np.int64)
    obj_ids : np.ndarray = np.asarray([1,], dtype=track_index_to_predict.dtype)
    obj_trajs_full : np.ndarray = np.concatenate([obj_trajs_past, obj_trajs_future], axis=1)

    tsamp_waymo-=tsamp_waymo[0]
    (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
                center_gt_trajs_mask, center_gt_final_valid_idx,
                track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = \
                    create_agent_data_for_center_objects(obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, tsamp_waymo[:num_timestamps], obj_types, obj_ids)
    ret_dict = {
        'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
        'obj_trajs': obj_trajs_data,
        'obj_trajs_mask': obj_trajs_mask,
        'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
        'obj_trajs_pos': obj_trajs_pos,
        'obj_trajs_last_pos': obj_trajs_last_pos,
        'obj_types': obj_types,
        'obj_ids': obj_ids,

        # 'center_objects_world': center_objects, 
        'center_objects_id': obj_ids.copy(),
        'center_objects_type': obj_types.copy(),

        'obj_trajs_future_state': obj_trajs_future_state,
        'obj_trajs_future_mask': obj_trajs_future_mask,
        'center_gt_trajs': center_gt_trajs,
        'center_gt_trajs_mask': center_gt_trajs_mask,
        'center_gt_final_valid_idx': center_gt_final_valid_idx,
        'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
    }

    local_points = np.concatenate([lb_points, rb_points], axis=0, dtype=current_position.dtype)
    local_polygon = np.concatenate([lb_points, np.flipud(rb_points), lb_points[0][None]], axis=0, dtype=current_position.dtype)
    local_directions = np.concatenate([lb_tangents, rb_tangents], axis=0, dtype=current_position.dtype)

    polylines : np.ndarray = 15.0*np.ones((1, local_points.shape[0], 7), dtype=current_position.dtype)
    polylines[0,:,0:3] = local_points
    polylines[0,:,3:6] = local_directions
    # polylines_squeeze = np.squeeze(polylines)

    # polylines : np.ndarray = sample["map_infos"]["all_polylines"][None]
    map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
        polylines, polyline_config
    )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

    ret_dict['map_polygon'] = local_polygon
    ret_dict['map_points'] = local_points
    ret_dict['map_polylines'] = map_polylines_data
    ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
    ret_dict['map_polylines_center'] = map_polylines_center

    return ret_dict
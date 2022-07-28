import os
import cv2
import torch
import numpy as np

from core.triangulation import triangular_parse_camera
from core.triangulation import triangular_re_id
from core.triangulation import triangular_process_with_identity

from core.dmae_handler import dmae_model_prepare
from core.dmae_handler import DMAEQueueController

from util.config import ShelfCfg, parse_args
from util.skel_painter import draw_skels2d_for_each_view


def data_check(path):
    assert os.path.exists(path), "Invalid path: {}".format(path)


if __name__ == '__main__':
    mocap_cfg = ShelfCfg(parse_args())
    # 0. data validity check
    data_check(mocap_cfg.data_root)
    data_check(mocap_cfg.img_root)
    data_check(mocap_cfg.model_path)
    data_check(mocap_cfg.camera_param_path)

    # 1. prepare data
    # 1.1 prepare the model
    mae_model = None
    mae_queue = None
    device = torch.device("cuda:0")
    if mocap_cfg.dmae_enable:
        mae_model = dmae_model_prepare(device,
                                       model_path=mocap_cfg.model_path)
        mae_queue = DMAEQueueController(debug=False, ch=mocap_cfg.queue_ch)
    # 1.2 prepare camera parameters
    camera_param_path = mocap_cfg.camera_param_path
    camera_num, camera_res, camera_proj, camera_RtKi, camera_pos = triangular_parse_camera(camera_param_path)
    # 1.3 prepare data list
    mocap_skel2d_data = np.load(mocap_cfg.skel2d_dict_path, allow_pickle=True).tolist()
    assert len(mocap_skel2d_data) == len(mocap_cfg.view_list), "Wrong view number: {}".format(len(mocap_skel2d_data))
    if not os.path.exists(mocap_cfg.out_folder):
        os.makedirs(mocap_cfg.out_folder)
    if mocap_cfg.snapshot_flag and not os.path.exists(mocap_cfg.snp_folder):
        os.makedirs(mocap_cfg.snp_folder)
    mocap_frame_num = len(mocap_skel2d_data[mocap_cfg.view_list[0]])
    mocap_frame_idx = []
    for dirs in os.listdir(os.path.join(mocap_cfg.img_root, mocap_cfg.view_list[0])):
        if mocap_cfg.img_suffix in dirs:
            mocap_frame_idx.append(dirs.split(".")[0])
    assert mocap_frame_num == len(mocap_frame_idx), "Wrong frame number: {}".format(mocap_frame_num)
    print("===\nSRC_path: {} with frame Num: {}\nDST_path: {}\nCamera num:{}\nD-MAE enable: {}\n===".format(
        mocap_cfg.data_root,
        mocap_frame_num,
        mocap_cfg.out_folder,
        camera_num,
        mocap_cfg.dmae_enable))
    # 2. process each frame
    pre_frame_idx = None
    pre_skel3ds_with_id = None
    for frameIdx in range(mocap_cfg.frame_start, mocap_cfg.frame_end):
        frame_imgs_path = []
        frame_skel_data = []
        for view_dir in mocap_cfg.view_list:
            frame_imgs_path.append(os.path.join(mocap_cfg.img_root, view_dir,
                                                mocap_frame_idx[frameIdx] + mocap_cfg.img_suffix))
            frame_skel_data.append(mocap_skel2d_data[view_dir][mocap_frame_idx[frameIdx]])
        skel3d_dst_path = os.path.join(mocap_cfg.out_folder, mocap_frame_idx[frameIdx] + ".npy")
        snapshot_dst_path = os.path.join(mocap_cfg.snp_folder, mocap_frame_idx[frameIdx] + ".jpg")
        # 2.0 jump processed frame to save time
        if os.path.exists(skel3d_dst_path) or os.path.exists(snapshot_dst_path):
            print("Frame {} jumped".format(frameIdx))
            pre_skel3ds_with_id = np.load(skel3d_dst_path, allow_pickle=True).tolist()
            if mocap_cfg.dmae_enable:
                pre_skel3ds_with_id = [item[0] for item in pre_skel3ds_with_id]
            else:
                pre_skel3ds_with_id = [item for item in pre_skel3ds_with_id]
            continue

        # 2.1 load OpenPose skeleton data
        skels_data = []
        filtered_skels = []
        for skel_dict in frame_skel_data:
            filtered_skel = []
            for skel_each in skel_dict:
                potential_skel = skel_each["filtered_skel"]
                prob_skel = potential_skel[:, 2]
                # we filter skeleton if the number of joint is invalid
                if np.sum(prob_skel == 0.0) > mocap_cfg.min_joint_num:
                    continue
                # we filter skeleton if the hip relative joint is invisible
                if prob_skel[8] == 0.0 or (prob_skel[9] == 0.0 and prob_skel[12] == 0.0) or prob_skel[1] == 0.0:
                    continue
                filtered_skel.append(potential_skel)
            filtered_skels.append(filtered_skel)  # view * frame_num, skel_num, joint_num, ch

        # 2.2 get identity for each skeleton on every view
        # 2.2.1 check frame index consistency
        cur_frame_idx = int(mocap_frame_idx[frameIdx])
        if pre_frame_idx is not None and cur_frame_idx != pre_frame_idx + 1:
            pre_skel3ds_with_id = None
            print("New scene appeared")
        skels_identity_map, skel_num = triangular_re_id(filtered_skels, camera_proj, camera_RtKi, camera_res,
                                                        camera_pos, view_path=frame_imgs_path,
                                                        pre_skel3d_with_id=pre_skel3ds_with_id)

        # 2.3 triangular
        print("Process Frame {}".format(frameIdx))
        skel3d_set_with_id, _ = triangular_process_with_identity(filtered_skels, skels_identity_map, camera_proj,
                                                                 camera_res, skel_num,
                                                                 min_view_skel=mocap_cfg.min_view_skel,
                                                                 min_view_joint=mocap_cfg.min_view_joint,
                                                                 filter_prob=mocap_cfg.filter_prob,
                                                                 model=mae_model,
                                                                 queue=mae_queue,
                                                                 ft=True)

        # 2.3.1 store current skel3d_with_id as pre_skel3d_with_id
        if mocap_cfg.dmae_enable:
            pre_skel3ds_with_id = [item[0] for item in skel3d_set_with_id]
        else:
            pre_skel3ds_with_id = skel3d_set_with_id
        pre_frame_idx = cur_frame_idx

        # 2.4 store skel3d data and snapshot (optional)
        if not len(skel3d_set_with_id):
            np.save(skel3d_dst_path, skel3d_set_with_id)
            continue
        if mocap_cfg.snapshot_flag:
            out = np.hstack(draw_skels2d_for_each_view(skel3d_set_with_id, camera_proj, frame_imgs_path))
            snapshot_out = cv2.resize(out, None, fx=mocap_cfg.snapshot_rescale_ratio,
                                      fy=mocap_cfg.snapshot_rescale_ratio)
            cv2.imwrite(snapshot_dst_path, snapshot_out)
        np.save(skel3d_dst_path, skel3d_set_with_id)

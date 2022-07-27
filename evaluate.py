import os
import cv2
import warnings
import numpy as np

from util.config import ShelfEvaCfg
from util.skel_utils import SkelConverter
from util.utils import util_re_id, util_evaluate_driver, util_evaluate_summary
from core.triangulation import triangular_parse_camera
from util.skel_painter import draw_shelf_for_each_view


def eval_snapshot(mocap_config, frameIdx, src_frame, gt_frame, camera_proj):
    frame_imgs_path = [os.path.join(mocap_cfg.img_root, item + "/{}{}".format(frameIdx, mocap_cfg.img_suffix))
                       for item in mocap_config.view_list]
    snapshot_dst_path = os.path.join(mocap_cfg.eval_snp_folder, frameIdx + ".jpg")

    canvas_out = draw_shelf_for_each_view(gt_frame, camera_proj, frame_imgs_path, type=0)
    canvas_out = draw_shelf_for_each_view(src_frame, camera_proj, frame_imgs_path, type=1, view_set=canvas_out)
    out = np.hstack(canvas_out)
    snapshot_out = cv2.resize(out, None, fx=mocap_cfg.snapshot_rescale_ratio, fy=mocap_cfg.snapshot_rescale_ratio)
    cv2.imwrite(snapshot_dst_path, snapshot_out)


if __name__ == '__main__':
    mocap_cfg = ShelfEvaCfg()
    mocap_skel_converter = SkelConverter(mocap_cfg)
    # 1. load GT data
    gt_data = np.load(mocap_cfg.gt_path, allow_pickle=True).tolist()  # frame X skel X joint X ch

    # 2. load Pred data
    pred_root = mocap_cfg.pred_root
    pred_data = []
    file_name = []
    for dirs in os.listdir(mocap_cfg.pred_root):
        if "npy" in dirs:
            pred_data.append(np.load(os.path.join(mocap_cfg.pred_root, dirs), allow_pickle=True).tolist())
            file_name.append(dirs.split(".")[0])

    # 2.1 check data validity
    if mocap_cfg.eval_snapshot_flag:
        if not os.path.exists(mocap_cfg.eval_snp_folder):
            os.makedirs(mocap_cfg.eval_snp_folder)
        _, camera_res, camera_proj, _, _ = triangular_parse_camera(mocap_cfg.camera_param_path)
    if len(pred_data) != len(gt_data):
        warn_msg = "Inconsistent data length: src {}, gt {}".format(len(pred_data), len(gt_data))
        warnings.warn(warn_msg)
    else:
        print("Load src data {}, gt data {}".format(len(pred_data), len(gt_data)))

    # 3. evaluate
    eval_metric = []
    for pred_frame_idx, (pred_frame_body25, gt_frame) in enumerate(zip(pred_data, gt_data)):
        print("Evaluate at Frame {}".format(pred_frame_idx))
        # 3.1 Skeleton type conversion
        gt_frame, pred_frame = mocap_skel_converter(gt_frame, pred_frame_body25)
        # 3.2 Re-id
        pred_frame_id_map = util_re_id(pred_frame, gt_frame, th=mocap_cfg.eval_reid_th)
        # 3.3 Evaluate
        eval_metric.append(util_evaluate_driver(pred_frame, gt_frame, pred_frame_id_map, pck_th=mocap_cfg.eval_pck_th))
        # 3.4 Snapshot (optional)
        if mocap_cfg.eval_snapshot_flag:
            eval_snapshot(mocap_cfg, file_name[pred_frame_idx], pred_frame, gt_frame, camera_proj)
    # 4. summary
    util_evaluate_summary(eval_metric, mocap_cfg.out_folder)

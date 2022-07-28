import copy
import numpy as np

from .skel_def import *
from .utils import util_norm


def util_convert_BODY21_to_SHELF14(src_skel3d, src_skel3d_pre=None, triangulate_th=0.05, face_y=0.125, face_z=0.145):
    dst_skel3d = [np.zeros((3, 1)) for i in range(15)]

    # new feature: data=[openpose(21), shelf(14)]
    if len(src_skel3d) == 2:
        fuse14 = [util_convert_BODY21_to_SHELF14_v2(src_skel3d,
                                                    triangulate_th=triangulate_th,
                                                    face_y=face_y,
                                                    face_z=face_z),
                  np.vstack((src_skel3d[1], (src_skel3d[1][2] + src_skel3d[1][3]) / 2))]
        return fuse14

    # note: triangulate_th > convergent's update_tolerance
    #       convergent's flag maybe False while we still think this result is valid
    # note: magic number = [False, 0.07] in src_skel3d means this joint is predicted
    #       due to current model, we cannot generate REye(15) and LEye(16)
    #       we use predicted joints as BOTTOM_HEAD(13) and TOP_HEAD(12)
    for src_jIdx, dst_jIdx in zip(convert_map_BODY21_SHELF14[0], convert_map_BODY21_SHELF14[1]):
        if src_skel3d[src_jIdx][2] <= triangulate_th and dst_jIdx >= 0:
            dst_skel3d[dst_jIdx] = src_skel3d[src_jIdx][1]
    if sum(dst_skel3d[12]) == 0 or sum(dst_skel3d[14]) == 0 or sum(dst_skel3d[8]) == 0 or sum(dst_skel3d[9]) == 0:
        # we can't have the main body part
        return dst_skel3d
    # get face direction
    face_dir = np.cross((dst_skel3d[12] - dst_skel3d[14]).T, (dst_skel3d[8] - dst_skel3d[9]).T)
    face_dir_normalized = face_dir / util_norm(face_dir)
    z_dir = np.zeros((3, 1))
    z_dir[2, 0] = 1.
    # calc BOTTOM_HEAD(13) and TOP_HEAD(12)
    # check validation, if invalid, we use the previous frame's result
    # otherwise, discard this joint
    ear_visible = True
    if src_skel3d[17][0]:
        src_ear_r = src_skel3d[17][1]
    elif src_skel3d_pre is not None and src_skel3d_pre[17][0]:
        src_ear_r = src_skel3d_pre[17][1]
    else:
        ear_visible = False
    if src_skel3d[18][0]:
        src_ear_l = src_skel3d[18][1]
    elif src_skel3d_pre is not None and src_skel3d_pre[18][0]:
        src_ear_l = src_skel3d_pre[18][1]
    else:
        ear_visible = False

    if ear_visible:
        head_center = (src_ear_r + src_ear_l) / 2.
    # otherwise, we use nose as the head center
    elif src_skel3d[0][0]:
        head_center = src_skel3d[0][1]
        ear_visible = True
    elif src_skel3d_pre is not None and src_skel3d_pre[0][0]:
        head_center = src_skel3d_pre[0][1]
        ear_visible = True

    if ear_visible:
        shoulder_center = (dst_skel3d[8] + dst_skel3d[9]) / 2.
        dst_skel3d[12] = shoulder_center + (head_center - shoulder_center) * 0.5
        dst_skel3d[13] = dst_skel3d[12] + face_dir_normalized.T * face_y + z_dir * face_z

    # magic number
    if src_skel3d[1][0] is False and src_skel3d[1][2] == 0.07:
        dst_skel3d[12] = src_skel3d[1][1]
    if src_skel3d[0][0] is False and src_skel3d[0][2] == 0.07:
        dst_skel3d[13] = src_skel3d[0][1]
    return dst_skel3d


def util_convert_BODY21_to_SHELF14_v2(src_skel3d, triangulate_th=0.05, face_y=0.125, face_z=0.145):
    dst_skel3d = [np.zeros((3, 1)) for i in range(15)]
    # note: triangulate_th > convergent's update_tolerance
    #       convergent's flag maybe False while we still think this result is valid
    # note: magic number = [False, 0.07] in src_skel3d means this joint is predicted
    #       due to current model, we cannot generate REye(15) and LEye(16)
    #       we use predicted joints as BOTTOM_HEAD(13) and TOP_HEAD(12)
    skel3d_21 = src_skel3d[0]
    skel3d_14 = src_skel3d[1]
    for src_jIdx, dst_jIdx in zip(convert_map_BODY21_SHELF14_v2[0], convert_map_BODY21_SHELF14_v2[1]):
        if skel3d_21[src_jIdx][2] <= triangulate_th:
            # magic number
            if skel3d_21[src_jIdx][2] == 0.07 and skel3d_21[src_jIdx][0] == False:
                if dst_jIdx > 13:
                    continue
                dst_skel3d[dst_jIdx] = skel3d_14[dst_jIdx].reshape(-1, 1)
            else:
                dst_skel3d[dst_jIdx] = skel3d_21[src_jIdx][1]
    # if joint dist is greater than th, we trust prediction instead of triangulation
    if np.all(skel3d_14 != 0.):
        for jIdx in range(len(dst_skel3d) - 1):
            jDist = np.average(np.sqrt(np.sum(np.power(dst_skel3d[jIdx] - skel3d_14[jIdx].reshape(-1, 1), 2))))
            if jDist > 1:
                dst_skel3d[jIdx] = skel3d_14[jIdx].reshape(-1, 1)
    # if we already predict dst_joint@0/1, there is no need to calc it by src_joint@17/18
    if (skel3d_21[17][2] == 0.07 and skel3d_21[17][0] == False) \
            or (skel3d_21[18][2] == 0.07 and skel3d_21[18][0] == False):
        dst_skel3d[12] = skel3d_14[12].reshape(-1, 1)
        dst_skel3d[13] = skel3d_14[13].reshape(-1, 1)
        return dst_skel3d
    # calc dst_joint@0/1
    if sum(dst_skel3d[12]) == 0 or sum(dst_skel3d[14]) == 0 or sum(dst_skel3d[8]) == 0 or sum(dst_skel3d[9]) == 0:
        # we can't have the main body part
        return dst_skel3d
    # get face direction
    face_dir = np.cross((dst_skel3d[12] - dst_skel3d[14]).T, (dst_skel3d[8] - dst_skel3d[9]).T)
    face_dir_normalized = face_dir / util_norm(face_dir)
    z_dir = np.zeros((3, 1))
    z_dir[2, 0] = 1.
    # calc BOTTOM_HEAD(13) and TOP_HEAD(12)
    ear_visible = True
    if skel3d_21[17][0]:
        src_ear_r = skel3d_21[17][1]
    else:
        ear_visible = False
    if skel3d_21[18][0]:
        src_ear_l = skel3d_21[18][1]
    else:
        ear_visible = False
    if ear_visible:
        head_center = (src_ear_r + src_ear_l) / 2.
    # otherwise, we use nose as the head center
    elif skel3d_21[0][0]:
        head_center = skel3d_21[0][1]
        ear_visible = True
    if ear_visible:
        shoulder_center = (dst_skel3d[8] + dst_skel3d[9]) / 2.
        dst_skel3d[12] = shoulder_center + (head_center - shoulder_center) * 0.5
        dst_skel3d[13] = dst_skel3d[12] + face_dir_normalized.T * face_y + z_dir * face_z

    return dst_skel3d


def util_convert_SHELF14_to_BODY21(src_pred, src_data, normalizer):
    dst_skel3d_14 = src_pred.copy()
    dst_skel3d_14 += normalizer
    dst_skel3d_14 = np.vstack((dst_skel3d_14, normalizer.reshape(1, -1)))
    dst_skel3d_21 = copy.deepcopy(src_data)

    # note: magic number = [False, 0.07] means this joint is predicted
    for jIdx, joint in enumerate(dst_skel3d_21):
        if joint[0] is False:
            jIdx_shelf = convert_map_BODY21_SHELF14[1][jIdx]
            if jIdx_shelf != -1:
                joint[1] = dst_skel3d_14[jIdx_shelf].reshape(-1, 1)
                joint[2] = 0.07
    return dst_skel3d_21


def util_convert_MAE21_to_BODY21(src_pred, src_data, normalizer, v2=True):
    if v2:
        mae_21 = src_pred.copy()
        mae_21 = mae_21[14:]
    else:
        mae_21 = src_pred.copy()
    mae_21 += normalizer
    dst_skel3d_21 = copy.deepcopy(src_data)

    # note: magic number = [False, 0.07] means this joint is predicted
    for jIdx, joint in enumerate(dst_skel3d_21):
        if joint[0] is False:
            joint[1] = mae_21[jIdx].reshape(-1, 1)
            joint[2] = 0.07
    return dst_skel3d_21


def util_convert_MAE21_to_SHELF14(src_pred, src_data, normalizer):
    mae_14 = src_pred.copy()
    mae_14 = mae_14[:14]
    mae_14 += normalizer
    return mae_14


def util_convert_PRED21_to_SKEL19(src_data):
    assert len(convert_map_PRED21_SKEL19) == 21
    dst_data = np.zeros((max(convert_map_BODY25_SKEL19) + 1, 3))
    # joint mapping
    if len(src_data) == 2:
        # v2 feature
        src_data = src_data[0]
    # delete 15th (REye), 16th (LEye)
    for jIdx, srcJ in enumerate(src_data):
        if convert_map_BODY25_SKEL19[jIdx] == -1:
            continue
        dst_data[convert_map_BODY25_SKEL19[jIdx]] = srcJ[1].squeeze()
    return dst_data


def util_convert_BODY25_to_SKEL19(src_data):
    assert len(convert_map_BODY25_SKEL19) == 25
    dst_data = np.zeros((max(convert_map_BODY25_SKEL19) + 1, 3))
    # joint mapping
    # delete 15th (REye), 16th (LEye), 20th, 21th, 23th, 24th, 5 joints
    for jIdx, srcJ in enumerate(src_data):
        if convert_map_BODY25_SKEL19[jIdx] == -1:
            continue
        dst_data[convert_map_BODY25_SKEL19[jIdx]] = srcJ
    return dst_data


def util_convert_BODY25_to_SKEL17(src_data):
    assert len(convert_map_BODY25_SKEL17) == 25
    dst_data = np.zeros((max(convert_map_BODY25_SKEL17) + 1, src_data.shape[1]))
    # joint mapping
    for jIdx, srcJ in enumerate(src_data):
        if convert_map_BODY25_SKEL17[jIdx] == -1:
            continue
        dst_data[convert_map_BODY25_SKEL17[jIdx]] = srcJ
    return dst_data


def util_convert_PRED17_to_SKEL17(src_data):
    assert False, "Not applicable"
    extra_joint = np.zeros((2, 3))
    # add mid-shoulder (17) and mid-hip (18)
    extra_joint[0] = (src_data[5] + src_data[6]) / 2.
    extra_joint[1] = (src_data[11] + src_data[12]) / 2.
    return np.vstack((src_data, extra_joint))


class SkelConverter:
    def __init__(self, mocap_config):
        self.config_name = mocap_config.__class__.__name__.lower()

    def __call__(self, gt_frame, pred_frame):
        if 'shelf' in self.config_name:
            pred_frame = [util_convert_BODY21_to_SHELF14(skel3d_body25, triangulate_th=0.1) for skel3d_body25 in
                          pred_frame]
        elif 'mvpose' in self.config_name:
            # covert from BODY21 to SKEL17
            gt_frame = [util_convert_BODY25_to_SKEL17(body25) for body25 in gt_frame]
            pred_frame = [util_convert_PRED17_to_SKEL17(pred17.T) for pred17 in pred_frame]
        elif '4da' in self.config_name:
            # covert from BODY25 to SKEL19
            gt_frame = [util_convert_BODY25_to_SKEL19(body25) for body25 in gt_frame]
            pred_frame = [pred19 for pred19 in pred_frame]
        elif 'coop' in self.config_name:
            # Convert from BODY21 to SKEL19
            gt_frame = [util_convert_BODY25_to_SKEL19(body25) for body25 in gt_frame]
            pred_frame = [util_convert_PRED21_to_SKEL19(pred21) for pred21 in pred_frame]
        return gt_frame, pred_frame

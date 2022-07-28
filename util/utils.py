import os
import cv2
import numpy as np

from .skel_def import eval_map_SHELF14_name
from .skel_def import eval_map_SHELF14, eval_map_SKEL17, eval_map_SKEL19


def util_norm(point: np.array):
    return np.sqrt(np.sum(np.square(point)))


def util_normalize(point: np.array):
    return point / util_norm(point)


def util_point2line_dist(point_a: np.array, point_b: np.array, line: np.array):
    return util_norm(np.cross((point_a - point_b), line))


def util_line2line_dist(point_a: np.array, line_a: np.array, point_b: np.array, line_b: np.array):
    if np.abs(np.dot(line_a, line_b)) < 1e-5:
        return util_point2line_dist(point_a, point_b, line_a)
    else:
        return np.abs(np.dot((point_a - point_b).T, util_normalize(np.cross(line_a, line_b))))


def util_visual_dist(view_path_a, point_a_s, point_a_e, view_path_b, point_b_s, point_b_e):
    def median_average(img, pad=20):
        b, g, r = cv2.split(img)
        median_b = np.median(b)
        median_g = np.median(g)
        median_r = np.median(r)
        mask = cv2.inRange(img, (median_b - pad, median_g - pad, median_r - pad),
                           (median_b + pad, median_g + pad, median_r + pad))
        mask_b = np.average(b[mask == 255])
        mask_g = np.average(g[mask == 255])
        mask_r = np.average(r[mask == 255])
        return np.array((mask_b, mask_g, mask_r)), mask

    def median_dist(img_0, img_1):
        bgr_0, mask_0 = median_average(img_0)
        bgr_1, mask_1 = median_average(img_1)
        dist = np.sqrt(np.sum(np.power(bgr_0 - bgr_1, 2))) / 255
        return dist, (mask_0, mask_1)

    def grab_bbox(img_path, line, padding=25):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        min_x = np.clip(int(np.min(line[:, 0]) * w - padding + 0.5), 0, w - 1)
        min_y = np.clip(int(np.min(line[:, 1]) * h + 0.5), 0, h - 1)
        max_x = np.clip(int(np.max(line[:, 0]) * w + padding + 0.5), 0, w - 1)
        max_y = np.clip(int(np.max(line[:, 1]) * h + 0.5), 0, h - 1)
        return img[min_y:max_y, min_x:max_x], [min_x, max_x, min_y, max_y]

    if not os.path.exists(view_path_a) or not os.path.exists(view_path_b):
        print("ERROR, no file found at {}, {}".format(view_path_a, view_path_b))
        return -1

    roi_img_a, _ = grab_bbox(view_path_a, np.vstack((point_a_s[:2], point_a_e[:2])))
    roi_img_b, _ = grab_bbox(view_path_b, np.vstack((point_b_s[:2], point_b_e[:2])))
    roi_img_a = cv2.resize(roi_img_a, (16, 16))
    roi_img_b = cv2.resize(roi_img_b, (16, 16))
    color_dist, color_mask = median_dist(roi_img_a, roi_img_b)
    return color_dist


def util_calc_pcp(skel3d_src, skel3d_gt, limb_map):
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim. {}-{}'.format(len(skel3d_src),
                                                                                                      len(skel3d_gt))
    pcp_mat = np.zeros((len(limb_map[0])))
    for limbIdx in range(len(limb_map[0])):
        jA_src = skel3d_src[limb_map[0][limbIdx]]
        jB_src = skel3d_src[limb_map[1][limbIdx]]
        # valid limb filter
        if np.all(jA_src == 0.) or np.all(jB_src == 0.):
            continue
        jA_gt = skel3d_gt[limb_map[0][limbIdx]]
        jB_gt = skel3d_gt[limb_map[1][limbIdx]]
        dist_a = util_norm(jA_src - jA_gt)
        dist_b = util_norm(jB_src - jB_gt)
        l = util_norm(jA_gt - jB_gt)
        if dist_a + dist_b < l:
            pcp_mat[limbIdx] = 1
    return pcp_mat


def util_calc_pck(skel3d_src, skel3d_gt, th=0.2):
    assert th <= 1., 'the threshold must be less than 1m. the unit is mm.'
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim.'
    joint_num = len(skel3d_gt)
    correct_joint = np.zeros(joint_num)
    detected_joint = np.zeros(joint_num)
    for jIdx, (j_src, j_gt) in enumerate(zip(skel3d_src, skel3d_gt)):
        # valid joint filter
        if np.all(j_src == 0.):
            # not detected
            continue
        dist = util_norm(j_src - j_gt)
        if dist < th:
            correct_joint[jIdx] = 1
        if np.sum(np.abs(j_src)) > 0:
            detected_joint[jIdx] = 1
    return correct_joint, detected_joint


def util_calc_mpjpe(skel3d_src, skel3d_gt):
    assert len(skel3d_src) == len(skel3d_gt), 'Input skeleton should have the same dim.'
    mpjpe = 0
    for j_src, j_gt in zip(skel3d_src, skel3d_gt):
        mpjpe += util_norm(j_src - j_gt)
    return mpjpe / len(skel3d_gt)


def util_re_id(frame_src, frame_gt, th=0.2):
    # return which skeletons in frame_src belongs to frame_gt
    # aka, from src to gt
    frame_src_id_map = np.ones(len(frame_src)) * -1
    for skel_src_id, skel_src in enumerate(frame_src):
        dist_list = []
        # v2 feature
        if len(skel_src) == 2:
            skel_src = skel_src[0]
        if isinstance(skel_src, list):
            if len(skel_src) == 0:
                continue
            skel_src = np.array(skel_src)[:, :, 0]
        for skel_gt in frame_gt:
            if skel_gt is None or not len(skel_gt):
                dist_list.append(1000.)
                continue
            # make sure src and gt are the same data structure
            mask = np.logical_and(np.sum(skel_src, axis=1) != 0, np.sum(skel_gt, axis=1) != 0)
            dist_list.append(np.average(np.sqrt(np.sum(np.power(skel_src[mask] - skel_gt[mask], 2), axis=1))))
        min_dist = min(dist_list)
        if min_dist < th:
            min_idx = dist_list.index(min_dist)  # find the closest gt id
            frame_src_id_map[skel_src_id] = min_idx
    return frame_src_id_map


def util_evaluate_driver(frame_src, frame_gt, frame_src_id_map, pck_th=0.2):
    # assert len(frame_gt) == 4, "GT dim error! {}".format(len(frame_gt))
    pcp_score = [[], [], [], []]
    pck_score = [[], [], [], []]
    mpjpe_score = [[], [], [], []]
    joint_num = 0
    log_id = []
    for skel_src, skel_gt_id in zip(frame_src, frame_src_id_map):
        skel_gt_id = int(skel_gt_id)
        # gt_id = -1, means mis-match
        if skel_gt_id == -1:
            continue
        skel_gt = frame_gt[skel_gt_id]
        joint_num = len(skel_gt)
        # make sure gt is not empty
        if not len(skel_gt):
            continue
        log_id.append(skel_gt_id)
        # make sure src and gt are the same data structure
        if not isinstance(skel_gt, list):
            skel_gt = skel_gt.tolist()
            skel_gt = [np.array(item).reshape(3, -1) for item in skel_gt]
        if not isinstance(skel_src, list):
            skel_src = skel_src.tolist()
            skel_src = [np.array(item).reshape(3, -1) for item in skel_src]
        # v2 feature
        if len(skel_src) == 2:
            pcp_score_shelf = util_calc_pcp(skel_src[0], skel_gt, eval_map_SHELF14)
            pcp_score_opps = util_calc_pcp([joint.reshape(-1, 1) for joint in skel_src[1]], skel_gt, eval_map_SHELF14)
            if np.sum(pcp_score_shelf) >= np.sum(pcp_score_opps):
                pcp_score[skel_gt_id].append(pcp_score_shelf)
            else:
                pcp_score[skel_gt_id].append(pcp_score_opps)
            pck_score[skel_gt_id].append(util_calc_pck(skel_src[0], skel_gt, pck_th))
            mpjpe_score[skel_gt_id].append(util_calc_mpjpe(skel_src[0], skel_gt))
        else:
            PCP_limb_map = None
            if len(skel_gt) == 15:
                PCP_limb_map = eval_map_SHELF14
            elif len(skel_gt) == 19:
                PCP_limb_map = eval_map_SKEL19
            elif len(skel_gt) == 17:
                PCP_limb_map = eval_map_SKEL17
            pcp_score[skel_gt_id].append(util_calc_pcp(skel_src, skel_gt, PCP_limb_map))
            pck_score[skel_gt_id].append(util_calc_pck(skel_src, skel_gt, pck_th))
            mpjpe_score[skel_gt_id].append(util_calc_mpjpe(skel_src, skel_gt))
    # check unpredictable result
    for gId, gt in enumerate(frame_gt):
        if gId not in log_id and len(gt):
            pcp_score[gId].append(np.zeros(len(PCP_limb_map[0])))
            tmp = np.zeros(joint_num)
            pck_score[gId].append((tmp, tmp))
            mpjpe_score[gId].append(0)
    return {
        "pcp": pcp_score,
        "pck": pck_score,
        "pck@th": pck_th,
        "mpjpe": mpjpe_score,
        "joint_num": joint_num
    }


def util_evaluate_summary(eval_metric, txt_dir):
    import logging, os
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(txt_dir, 'eval.log'), 'a'))
    print = logger.info

    eps = np.finfo(float).eps
    eval_frame_num = len(eval_metric)
    joint_num = eval_metric[0]['joint_num']
    print("=== Evaluation summary ===")
    print("- Frame num: {}".format(eval_frame_num))
    print("- Skel num: 4")
    print("- Joint num: {} (include mid-hip)".format(joint_num))

    PCP_metric = np.zeros((4, len(eval_map_SHELF14_name)))
    PCP_valid_frame = np.ones((4, 1)) * eps
    PCK_r_metric = np.ones((4, joint_num)) * eps
    PCK_p_metric = np.ones((4, joint_num)) * eps
    PCK_valid_frame = np.ones((4, 1)) * eps
    MPJPE_metric = np.zeros(4)
    MPJPE_valid_frame = np.ones((4, 1)) * eps
    for frame in eval_metric:
        # PCP
        for pIdx, pcp_list in enumerate(frame["pcp"]):
            if len(pcp_list):
                PCP_valid_frame[pIdx] += 1
                pcp_list = pcp_list[0]
                for jIdx, pcp_j in enumerate(pcp_list):
                    PCP_metric[pIdx, jIdx] += pcp_j
        # PCK
        for pIdx, pck_list in enumerate(frame["pck"]):
            if len(pck_list):
                PCK_valid_frame[pIdx] += 1
                pck_list = pck_list[0]
                pck_r_list = pck_list[0]
                pck_p_list = pck_list[1]
                for jIdx, (pck_r, pck_p) in enumerate(zip(pck_r_list, pck_p_list)):
                    PCK_r_metric[pIdx, jIdx] += pck_r
                    PCK_p_metric[pIdx, jIdx] += pck_p
        # MPJPE
        for pIdx, mpjpe_list in enumerate(frame["mpjpe"]):
            if len(mpjpe_list):
                MPJPE_metric[pIdx] += mpjpe_list[0]
                MPJPE_valid_frame[pIdx] += 1

    # average
    PCP_avg = PCP_metric / PCP_valid_frame
    PCP_acg3 = []
    precision_avg = PCK_r_metric / PCK_p_metric
    recall_avg = PCK_r_metric / PCK_valid_frame
    MPJPE_avg = MPJPE_metric / MPJPE_valid_frame.squeeze(axis=1)
    print("=== PCP ===")
    for pIdx, person in enumerate(PCP_metric):
        print("\tP {} ===".format(pIdx))
        for lIdx, limb_score in enumerate(person):
            limb_name = eval_map_SHELF14_name[lIdx]
            print("{}: {}/{}, {}".format(limb_name, int(limb_score), int(PCP_valid_frame[pIdx]), PCP_avg[pIdx, lIdx]))
        PCP_acg3.append(np.average(PCP_avg[pIdx]))
        print("\tAverage: {}".format(np.average(PCP_avg[pIdx])))
    print("=== PCP AVG P1-3 ===")
    print("\tAverage: {}".format(np.average(PCP_acg3[:3])))
    print("=== PCK@{} ===".format(eval_metric[0]["pck@th"]))
    for pIdx, (person_p, person_r) in enumerate(zip(precision_avg, recall_avg)):
        print("\tP {} ===".format(pIdx))
        print("\tAverage Precision: {}".format(np.average(person_p)))
        print("\tAverage Recall: {}".format(np.average(person_r)))
    return

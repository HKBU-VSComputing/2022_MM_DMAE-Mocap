from itertools import combinations, product

import os
import cv2
import numpy as np
import scipy.linalg
import copy

from util.utils import util_norm, util_line2line_dist, util_visual_dist
from .dmae_handler import dmae_data_preprocess, dmae_data_postprocess, dmae_model_process


def triangular_parse_camera(src_path, rectify=False):
    assert os.path.exists(src_path), "No file: {}".format(src_path)
    if 'pickle' in src_path:
        param = np.load(src_path, allow_pickle=True)
    else:
        param = np.load(src_path, allow_pickle=True).tolist()
    view_num = len(param)
    res = param[0]["res"]
    proj_mat = []
    RtKi_mat = []
    pos_mat = []
    for camIdx, cam in enumerate(param):
        proj = cam["P"]
        K = np.array(cam["K"])
        RtKi = np.dot(np.array(cam["R"]).T, np.linalg.inv(K))
        proj_mat.append(proj)
        RtKi_mat.append(RtKi)
        pos_mat.append(cam["Pos"])
    proj_mat = np.array(proj_mat)
    return view_num, res, proj_mat, RtKi_mat, pos_mat


def triangular_solve(max_iter_time=20, update_tolerance=0.0001, regular_term=0.0001, point=None, proj_mat=None,
                     view_num=5, filter_prob=0.2):
    point = point.T
    assert view_num == proj_mat.shape[0]
    pos = np.zeros((4, 1))
    pos[3] = 1
    convergent = False

    # filter invalid point
    prob = point[2, :] > filter_prob

    # at least 2 views have seen the skeleton
    # which can perform triangulate
    if sum(prob) < 2:
        return False, np.zeros((3, 1)), 1

    loss = 100.
    for i in range(max_iter_time):
        ATA = np.identity(3) * regular_term
        ATb = np.zeros((3, 1))
        for view in range(view_num):
            # visible point
            if prob[view] > 0:
                current_proj = proj_mat[view]  # (3, 4)
                xyz = np.dot(current_proj, pos)
                ##
                jacobi = np.zeros((2, 3))
                jacobi[0, 0] = 1. / xyz[2]
                jacobi[0, 2] = -xyz[0] / pow(xyz[2], 2)
                jacobi[1, 1] = 1. / xyz[2]
                jacobi[1, 2] = -xyz[1] / pow(xyz[2], 2)
                ##
                jacobi = np.dot(jacobi, current_proj[:, :3])
                w = point[2, view]  # prob
                ATA += w * np.dot(jacobi.T, jacobi)
                out = (point[:2, view] - xyz[:2, 0] / xyz[2, 0]).reshape(-1, 1)
                ATb += w * np.dot(jacobi.T, out)
        delta = scipy.linalg.solve(ATA, ATb)
        loss = np.linalg.norm(delta)
        if loss < update_tolerance:
            convergent = True
            break
        else:
            pos[:3] += delta
    return convergent, pos[:3], loss


def triangular_process(skel2d_all, camera_proj, camera_res, strict=False, min_view_num=3, filter_prob=0.2):
    assert len(skel2d_all) == len(camera_proj)
    camera_num, joint_num, _ = skel2d_all.shape
    skel3d = []
    for joint_idx in range(joint_num):
        # process each joint in all view
        joint_mat = skel2d_all[:, joint_idx, :]
        if strict:
            # we only triangulate joint which has been seen by all view
            if np.any(joint_mat[:, 2] == 0.):
                skel3d.append((False, np.zeros((3, 1)), 100.))
                continue
        # or we only triangulate joint with the limited view number
        elif min_view_num > 0:
            if np.sum(joint_mat[:, 2] != 0) < min_view_num:
                skel3d.append([False, np.zeros((3, 1)), 100.])
                continue
        # calc back to pixel coordinate
        joint_mat[:, 0] *= (camera_res[0] - 1)
        joint_mat[:, 1] *= (camera_res[1] - 1)
        # solve
        convergent, joint_converged, loss = triangular_solve(point=joint_mat, proj_mat=camera_proj,
                                                             view_num=camera_num, filter_prob=filter_prob)
        skel3d.append([convergent, joint_converged, loss])
    return skel3d


def triangular_grab_boundingbox(skel2d_set, img_path, padding=20):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    boundingbox_img = []
    boundingbox_roi = []
    for skel_each in skel2d_set:
        skel_valid = skel_each[skel_each[:, 2] > 0]
        min_x = np.clip(int(np.min(skel_valid[:, 0]) * w - padding + 0.5), 0, w - 1)
        min_y = np.clip(int(np.min(skel_valid[:, 1]) * h - padding + 0.5), 0, h - 1)
        max_x = np.clip(int(np.max(skel_valid[:, 0]) * w + padding + 0.5), 0, w - 1)
        max_y = np.clip(int(np.max(skel_valid[:, 1]) * h + padding + 0.5), 0, h - 1)
        boundingbox_roi.append([min_x, max_x, min_y, max_y])
        roi_img = img[min_y:max_y, min_x:max_x]
        boundingbox_img.append(roi_img)
    return boundingbox_img, boundingbox_roi


def triangular_calc_similarity(img_a, img_b, size=(128, 128)):
    def pearson(image1, image2):
        X = np.vstack([image1, image2])
        return np.corrcoef(X)[0][1]

    image1 = cv2.resize(img_a, size)
    image2 = cv2.resize(img_b, size)
    image1 = np.asarray(image1).flatten()
    image2 = np.asarray(image2).flatten()
    return pearson(image1, image2)


def triangular_reprojection(skel3d, proj_mat, camera_res, triangulate_th=0.05):
    skel3d_data = np.ones((len(skel3d), 4))
    skel3d_data[:, :3] = np.array([item[1].squeeze() for item in skel3d])
    skel3d_loss = np.array([item[2] for item in skel3d])
    skel2d_set = []
    for idx, proj_each in enumerate(proj_mat):
        skel2d_each_view = []
        # let's reproject the 3d joint back to 2d view
        joint2d = np.dot(proj_each, skel3d_data.T).T
        joint2d_normalized_x = joint2d[:, 0] / joint2d[:, 2]
        joint2d_normalized_y = joint2d[:, 1] / joint2d[:, 2]
        # filter with loss to get valid joint
        for jIdx in range(len(skel3d_loss)):
            if skel3d_loss[jIdx] >= triangulate_th:
                skel2d_each_view.append((0., 0.))
                continue
            j2d_x = int(joint2d_normalized_x[jIdx] + 0.5) / (camera_res[0] - 1)
            j2d_y = int(joint2d_normalized_y[jIdx] + 0.5) / (camera_res[1] - 1)
            skel2d_each_view.append((j2d_x, j2d_y))
        skel2d_set.append(skel2d_each_view)
    return np.array(skel2d_set)


def triangular_calc_skel_similarity(skel2d_matching_set, skel2d_set):
    skel2d_set = skel2d_set.swapaxes(0, 1)[:, :, :, :2]
    euclidean_dist = np.average(np.sqrt(np.sum(np.square(skel2d_set - skel2d_matching_set), axis=3)), axis=2)
    skel2d_matching_id = np.argmin(euclidean_dist, axis=0)
    return skel2d_matching_id


def triangular_matching_skels(skel2d_set, imgs_path, camera_proj, camera_res):
    # matching skels
    skel3d_matching_set = []
    # 1. grab 2 views
    for (view_idx_a, skel_a), (view_idx_b, skel_b) in combinations(enumerate(skel2d_set), 2):
        img_path_a, img_path_b = imgs_path[view_idx_a], imgs_path[view_idx_b]
        camera_proj_pair = np.stack((camera_proj[view_idx_a], camera_proj[view_idx_b]))
        # 2. get skeleton's bounding box
        roi_img_a, roi_set_a = triangular_grab_boundingbox(skel_a, img_path_a)
        roi_img_b, roi_set_b = triangular_grab_boundingbox(skel_b, img_path_b)
        if len(roi_set_a) != len(roi_set_b):
            # two selected views have inconsistent skel number
            print("next 2 views!")
            continue
        # 3. calculate the similarity
        roi_similarity = []
        for skel_roi_a, skel_roi_b in product(roi_img_a, roi_img_b):
            roi_similarity.append(triangular_calc_similarity(skel_roi_a, skel_roi_b))
        roi_similarity = np.array(roi_similarity).reshape(len(roi_img_a), -1)
        # waiting an optimize algorithm
        roi_id_select = np.argmax(roi_similarity, axis=1)
        if sum(roi_id_select) != np.sum(range(len(roi_img_a))):
            print("next 2 views!")
            continue
        # 4. view-a vs view-b
        #    for the same skel in each view
        for skel_id_a, skel_id_b in enumerate(roi_id_select):
            same_skel_pair = np.stack((skel_a[skel_id_a], skel_b[skel_id_b]))
            # get the triangulate result
            skel3d_matching_set.append(triangular_process(same_skel_pair, camera_proj_pair, camera_res, min_view_num=0))
        # now we get 3d skeletons with identity triangulated by only 2 views
        # then we assign each view's skel2d's identity by these imprecise skel3d reprojections
        break
    # 5. reproject to each view
    skel2d_matching_id_set = []
    for skel3d_matching in skel3d_matching_set:
        skel2d_matching_set = triangular_reprojection(skel3d_matching, camera_proj, camera_res)
        skel2d_matching_id_set.append(triangular_calc_skel_similarity(skel2d_matching_set, skel2d_set))
    return np.array(skel2d_matching_id_set)


def triangular_process_with_identity(skel2d_all, skel_identity, camera_proj, camera_res, skel_num,
                                     min_view_skel=3, min_view_joint=0, filter_prob=0.2,
                                     model=None, queue=None,
                                     ft=False,
                                     panoptic=False,
                                     cuda_device="cuda:0"):
    assert len(skel2d_all) == len(camera_proj)

    skel3d_all_with_id = []
    if panoptic:
        skel3d_all_with_id_panoptic = []
    skel2d_all_with_id = [[] for i in range(skel_num)]  # person, view, joint, x, y, prob
    camera_proj_with_id = [[] for i in range(skel_num)]  # person, camera_proj
    for pId in range(skel_num):
        for viewIdx, skel2d_map in enumerate(skel_identity):
            for skelIdx, skelId in enumerate(skel2d_map):
                if skelId == pId and skelIdx < len(skel2d_all[viewIdx]):
                    skel2d_all_with_id[pId].append(skel2d_all[viewIdx][skelIdx])
                    camera_proj_with_id[pId].append(camera_proj[viewIdx])

    for id, (skel_with_id, proj_with_id) in enumerate(zip(skel2d_all_with_id, camera_proj_with_id)):
        # if visible view is less than min_view_skel, we abort triangular
        if min_view_skel > 0 and len(skel_with_id) < min_view_skel:
            continue
        if not len(skel_with_id):
            continue
        skel3d = triangular_process(np.array(skel_with_id), np.array(proj_with_id), camera_res,
                                    min_view_num=min_view_joint, filter_prob=filter_prob)
        # dmae inpainting
        if queue is not None and model is not None:
            # D-MAE
            # 1. find the missing part and prepare data
            # mask_data: mask_idx, unmask_idx
            # only for panoptic data ##############
            if panoptic:
                panoptic_skel3d = copy.deepcopy(skel3d)
                panoptic_skel3d = (panoptic_skel3d, np.zeros((14, 3)))
                for joint in skel3d:
                    # convert # y to z
                    #         # z = -z
                    #         # x, y, z * 10
                    joint_new = np.zeros((3, 1))
                    joint_new[0] = joint[1][0]
                    joint_new[1] = joint[1][2]
                    joint_new[2] = -joint[1][1]
                    joint_new *= 10
                    joint[1] = joint_new
            #######################################
            skel3d_14, mid_hip, mask_ratio, mask_data = dmae_data_preprocess(skel3d, v2=ft, v3=not ft)
            queue.append(skel3d_14, mid_hip, mask_data)
            if mask_ratio > 0:
                print("Mask ratio: {} for P{}".format(mask_ratio, id))
                # 2. complete the missing part
                data_in_skel, data_in_hip, data_mask = queue.fetch(skel3d_14, mid_hip)
                model_pred_35 = dmae_model_process(model, data_in_skel, data_mask, cuda_device, dim=queue.length)
                skel3d = dmae_data_postprocess(model_pred_35, skel3d, mid_hip, v2=ft, v3=not ft)
            elif ft:
                skel3d = (skel3d, np.zeros((14, 3)))

        skel3d_all_with_id.append(skel3d)
        if panoptic:
            skel3d_all_with_id_panoptic.append(panoptic_skel3d)
    if panoptic:
        return skel3d_all_with_id_panoptic, skel3d_all_with_id
    else:
        return skel3d_all_with_id, None


def triangular_ray_cast(point, RtKi, res):
    # calc back to pixel coordinate
    point[0] *= (res[0] - 1)
    point[1] *= (res[1] - 1)
    point[2] = 1
    ray = np.dot(-RtKi, point)
    ray_norm = util_norm(ray)
    ray /= ray_norm
    return ray


def triangular_re_id(skel2d_all, camera_proj, camera_RtKi, camera_res, camera_pos, max_epi_dist=0.15, normalized=True,
                     confidence=False, view_path=None,
                     pre_skel3d_with_id=None):
    assert len(skel2d_all) == len(camera_proj), "Inconsistent input skel2d and camera projection matrix."
    view_num = len(camera_proj)

    # 0. utilize previous skel3d with identity
    skel2d_all_mask = []
    pre_skel_assist = False
    if pre_skel3d_with_id is not None and len(pre_skel3d_with_id):
        pre_skel_id_map = np.full((len(pre_skel3d_with_id), view_num), -1)
        pre_skel_dist_map = np.full((len(pre_skel3d_with_id), view_num), -1.)
        for pre_skel_id, pre_skel3d in enumerate(pre_skel3d_with_id):
            # reproject back to 2d
            pre_skel2d_set = triangular_reprojection(pre_skel3d, camera_proj, camera_res)
            for view_idx, (pre_skel2d, cur_skel2d_list) in enumerate(zip(pre_skel2d_set, skel2d_all)):
                dist_map = np.full(len(cur_skel2d_list), 100.)
                for pIdx in range(len(cur_skel2d_list)):
                    mask = (cur_skel2d_list[pIdx][:, 2] > 0.) == (pre_skel2d[:, 1] > 0.)
                    dist = cur_skel2d_list[pIdx][mask, :2] - pre_skel2d[mask]
                    dist = np.average(np.sqrt(np.sum(np.square(dist), axis=1)), axis=0)
                    dist_map[pIdx] = dist
                if np.any(dist_map < 0.1):
                    closest_id = np.argmin(dist_map)
                    pre_skel_id_map[pre_skel_id][view_idx] = closest_id
                    pre_skel_dist_map[pre_skel_id][view_idx] = dist_map[closest_id]
        # check all previous skeletons have been assigned correctly
        for skel_a_idx in range(0, len(pre_skel_id_map) - 1):
            for skel_b_idx in range(skel_a_idx + 1, len(pre_skel_id_map)):
                skel_id_diff = pre_skel_id_map[skel_a_idx] - pre_skel_id_map[skel_b_idx]
                incorrect_idx = np.where(skel_id_diff == 0)[0]
                if not len(incorrect_idx):
                    continue
                incorrect_idx = incorrect_idx[np.where(pre_skel_id_map[skel_a_idx][incorrect_idx] != -1)]
                correct_idx = np.full((2, len(incorrect_idx)), -1)
                out = np.array((pre_skel_dist_map[skel_a_idx], pre_skel_dist_map[skel_b_idx]))
                out = np.argmin(out, axis=0)[incorrect_idx]
                for i, j in enumerate(out):
                    correct_idx[j, i] = pre_skel_id_map[skel_a_idx][incorrect_idx][i]
                assert correct_idx.shape[0] == 2
                pre_skel_id_map[skel_a_idx, incorrect_idx] = correct_idx[0]
                pre_skel_id_map[skel_b_idx, incorrect_idx] = correct_idx[1]
        # filter already assigned skel2d in each view
        for mask, view in zip(pre_skel_id_map.T, skel2d_all):
            view_mask = np.zeros(len(view))
            for maskId in mask:
                if maskId != -1:
                    view_mask[maskId] = 1
            skel2d_all_mask.append(view_mask)
        pre_skel_assist = True

    cur_skel3d_num = max([len(view) for view in skel2d_all])
    if cur_skel3d_num == 0:
        return [], cur_skel3d_num

    cur_skel3d_id_mat = -np.ones((view_num, view_num, cur_skel3d_num), dtype=np.int64)
    cur_skel3d_id_map = -np.ones((view_num, cur_skel3d_num), dtype=np.int64)
    # we assume that at least 3 views can see the same skeleton to get accurate triangular result
    # 1. let's calc each mid-hip's ray for each skeleton in every view
    cur_hip_ray_map = [[] for i in range(view_num)]
    for viewIdx, view in enumerate(skel2d_all):
        for skel2d_idx, skel2d in enumerate(view):
            # filter previous skel
            if pre_skel_assist and skel2d_all_mask[viewIdx][skel2d_idx] == 1:
                cur_hip_ray_map[viewIdx].append(np.zeros((1, 3)))
                continue
            mid_hip = copy.deepcopy(skel2d[8])
            assert mid_hip[2] > 0. and skel2d[1][2] > 0., "Filter error"
            # calc the ray cast for corresponding view/camera
            cur_hip_ray_map[viewIdx].append(triangular_ray_cast(mid_hip, camera_RtKi[viewIdx], camera_res))
    # 2. calc the similarity (epipolar distance & appearance distance)
    #    for 2 views/cameras each pair
    mid_hip_sim_dist = [[[] for j in range(view_num)] for i in range(view_num)]
    for viewIdx_a in range(0, view_num - 1):
        for viewIdx_b in range(viewIdx_a + 1, view_num):
            hip_num_a = len(cur_hip_ray_map[viewIdx_a])
            hip_num_b = len(cur_hip_ray_map[viewIdx_b])
            if hip_num_a == 0 or hip_num_b == 0:
                continue
            ray_dist_mat = np.zeros((hip_num_a, hip_num_b))
            color_dist_mat = np.zeros((hip_num_a, hip_num_b))
            for idx_a, hip_a in enumerate(cur_hip_ray_map[viewIdx_a]):
                for idx_b, hip_b in enumerate(cur_hip_ray_map[viewIdx_b]):
                    # filter previous skel
                    if pre_skel_assist and (
                            skel2d_all_mask[viewIdx_a][idx_a] == 1 or skel2d_all_mask[viewIdx_b][idx_b] == 1):
                        continue
                    ray_dist = util_line2line_dist(camera_pos[viewIdx_a],  # camera A
                                                   hip_a,
                                                   camera_pos[viewIdx_b],  # camera B
                                                   hip_b)
                    if view_path is not None:
                        visual_dist = util_visual_dist(view_path[viewIdx_a],  # camera A
                                                       skel2d_all[viewIdx_a][idx_a][1],  # spine
                                                       skel2d_all[viewIdx_a][idx_a][8],
                                                       view_path[viewIdx_b],  # camera B
                                                       skel2d_all[viewIdx_b][idx_b][1],  # spine
                                                       skel2d_all[viewIdx_b][idx_b][8])
                    if ray_dist < max_epi_dist:
                        ray_dist_mat[idx_a, idx_b] = 1. - ray_dist / max_epi_dist
                    if view_path is not None:
                        color_dist_mat[idx_a, idx_b] = visual_dist
            if normalized:
                def norm_mat(dist_mat):
                    row_len = np.clip(np.sum(dist_mat, axis=1), 1., np.inf)
                    col_len = np.clip(np.sum(dist_mat, axis=0), 1., np.inf)
                    for rowIdx in range(len(row_len)):
                        dist_mat[rowIdx, :] /= row_len[rowIdx]
                    for colIdx in range(len(col_len)):
                        dist_mat[:, colIdx] /= col_len[colIdx]
                    return dist_mat

                ray_dist_mat = norm_mat(ray_dist_mat)
                color_dist_mat = norm_mat(color_dist_mat)

            mid_hip_sim_dist[viewIdx_a][viewIdx_b] = ray_dist_mat - color_dist_mat
            if confidence:
                indices = mid_hip_sim_dist[viewIdx_a][viewIdx_b] == np.max(mid_hip_sim_dist[viewIdx_a][viewIdx_b],
                                                                           axis=0)
                mid_hip_sim_dist[viewIdx_a][viewIdx_b][indices == False] = 0.

    # 3. according to similarity map, we do re-id
    for viewIdx_a in range(0, view_num - 1):
        for viewIdx_b in range(viewIdx_a + 1, view_num):
            ray_dist_mat = mid_hip_sim_dist[viewIdx_a][viewIdx_b]
            # 3.1 Simply max each row, or
            # 3.2 Hungarian search
            if len(ray_dist_mat):
                hungarian_id_mat = triangular_hungarian_search(ray_dist_mat)
                if hungarian_id_mat is None:
                    continue
                cur_skel3d_id_mat[viewIdx_a, viewIdx_b, hungarian_id_mat[0]] = hungarian_id_mat[1]

    # 4. assign identity
    # pre-skel3d identity assistant
    inference_view_idx = 0
    if pre_skel_assist:
        for viewIdx in range(view_num):
            for skel_id in range(len(pre_skel_id_map)):
                pre_skel_id = pre_skel_id_map[skel_id][viewIdx]
                if pre_skel_id == -1:
                    continue
                cur_skel3d_id_map[viewIdx][pre_skel_id] = skel_id
        inference_view_idx = np.argmax(np.sum(cur_skel3d_id_map >= 0, axis=1))
        inference_view = cur_skel3d_id_map[inference_view_idx]
        for pIdx in range(cur_skel3d_num):
            if pIdx not in inference_view:
                tmp_idx = np.where(inference_view == -1)[0]
                if len(tmp_idx) == 0:
                    continue
                cur_skel3d_id_map[inference_view_idx][tmp_idx[0]] = pIdx
    else:
        # we assume the first skeleton in first view is the "First" man, then the "Second" man etc.
        for pIdx in range(cur_skel3d_num):
            if cur_skel3d_id_map[0][pIdx] == -1:
                cur_skel3d_id_map[0][pIdx] = pIdx
    # forward propagation
    for viewIdx_a in range(inference_view_idx, view_num - 1):
        for viewIdx_b in range(viewIdx_a + 1, view_num):
            # fetch skel id from a to b
            for pIdx in range(cur_skel3d_num):
                pIdx_view_b_in_view_a = cur_skel3d_id_mat[viewIdx_a][viewIdx_b][pIdx]
                if pIdx_view_b_in_view_a != -1:
                    cur_skel3d_id_map[viewIdx_b][pIdx_view_b_in_view_a] = cur_skel3d_id_map[viewIdx_a][pIdx]
    # backward propagation
    return cur_skel3d_id_map, cur_skel3d_num


def triangular_hungarian_search(src_mat: np.array):
    from scipy.optimize import linear_sum_assignment

    if np.all(src_mat <= 0):
        return
    row_ind, col_ind = linear_sum_assignment(src_mat, maximize=True)
    # check the invalid task
    # when invalid person number is greater than task num,
    # that means an invalid person has been assigned to a task
    # we need to filter it
    invalid_person_check = src_mat[row_ind, col_ind]
    invalid_list = []
    for pIdx, person in enumerate(invalid_person_check):
        if person <= 0:
            invalid_list.append(pIdx)
    if len(invalid_list):
        row_ind = np.delete(row_ind, invalid_list, axis=0)
        col_ind = np.delete(col_ind, invalid_list, axis=0)
    return (row_ind, col_ind)

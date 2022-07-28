import cv2
import numpy as np


def draw_skel_reprojection(skel3d, proj_mat, views_path, triangulate_th=0.1, view_set=None, limb_map=None, v2=False):
    assert len(proj_mat) == len(views_path)

    if v2:
        ##################################################
        # new feat: draw predicted shelf14
        # skel3d = (body21, shelf14)
        shelf14_pred = skel3d[1]
        shelf14_flag = np.all(shelf14_pred == 0., axis=1)
        shelf14_pred = np.hstack((shelf14_pred, np.ones((len(shelf14_pred), 1))))
        skel3d = skel3d[0]
        ##################################################
    skel3d_data = np.ones((len(skel3d), 4))
    skel3d_data[:, :3] = np.array([item[1].squeeze() for item in skel3d])
    skel3d_loss = np.array([item[2] for item in skel3d])
    skel3d_magic = np.array([(item[0] == False) and (item[2] == 0.07) for item in skel3d])
    view_img_set = []

    for idx, (proj_each, view_path_each) in enumerate(zip(proj_mat, views_path)):
        if view_set is not None and len(view_set):
            view_img = view_set[idx]
        else:
            view_img = cv2.imread(view_path_each)

        # let's reproject the 3d joint back to 2d view
        joint2d = np.dot(proj_each, skel3d_data.T).T
        joint2d_normalized_x = joint2d[:, 0] / joint2d[:, 2]
        joint2d_normalized_y = joint2d[:, 1] / joint2d[:, 2]
        # filter with loss to get valid joint
        j2d_x_list, j2d_y_list = [], []
        for jIdx in range(len(skel3d_loss)):
            if skel3d_loss[jIdx] >= triangulate_th:
                j2d_x_list.append(0.)
                j2d_y_list.append(0.)
                continue
            j2d_x = int(joint2d_normalized_x[jIdx] + 0.5)
            j2d_y = int(joint2d_normalized_y[jIdx] + 0.5)
            j2d_x_list.append(j2d_x)
            j2d_y_list.append(j2d_y)
            cv2.circle(view_img, (j2d_x, j2d_y), 2, (255, 0, 0), thickness=2, lineType=8, shift=0)
        # draw limb
        if limb_map is not None and len(limb_map):
            for limb in limb_map:
                j2d_a = (j2d_x_list[limb[0]], j2d_y_list[limb[0]])
                j2d_b = (j2d_x_list[limb[1]], j2d_y_list[limb[1]])
                color = (0, 255, 0)
                if skel3d_magic[limb[0]] or skel3d_magic[limb[1]]:
                    color = (0, 0, 255)
                if sum(j2d_a) > 0 and sum(j2d_b) > 0:
                    cv2.line(view_img, j2d_a, j2d_b, color, 1, 4)
        if v2:
            ##################################################
            # new feat: draw predicted shelf14
            from util import limb_map_SHELF14
            shelf_limb_map = np.array(limb_map_SHELF14).T
            joint2d = np.dot(proj_each, shelf14_pred.T).T
            joint2d_normalized_x = joint2d[:, 0] / joint2d[:, 2]
            joint2d_normalized_y = joint2d[:, 1] / joint2d[:, 2]
            # filter with loss to get valid joint
            j2d_x_list, j2d_y_list = [], []
            for jIdx in range(len(shelf14_flag)):
                if shelf14_flag[jIdx]:
                    j2d_x_list.append(0.)
                    j2d_y_list.append(0.)
                    continue
                j2d_x = int(joint2d_normalized_x[jIdx] + 0.5)
                j2d_y = int(joint2d_normalized_y[jIdx] + 0.5)
                j2d_x_list.append(j2d_x)
                j2d_y_list.append(j2d_y)
                cv2.circle(view_img, (j2d_x, j2d_y), 2, (255, 255, 255), thickness=2, lineType=8, shift=0)
            # draw limb
            for limb in shelf_limb_map:
                j2d_a = (j2d_x_list[limb[0]], j2d_y_list[limb[0]])
                j2d_b = (j2d_x_list[limb[1]], j2d_y_list[limb[1]])
                if sum(j2d_a) > 0 and sum(j2d_b) > 0:
                    cv2.line(view_img, j2d_a, j2d_b, (255, 255, 255), 1, 4)
            ##################################################
        view_img_set.append(view_img)
    return view_img_set


def draw_skels2d_for_each_view(skel3d_set, proj_mat, views_path, triangulate_th=0.1, v2=True):
    skel2d_view_set = []
    limb_map = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                         [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 20, 13, 14, 19, 17, 18]]).T
    limb_map = limb_map.tolist()
    for skel3d in skel3d_set:
        if not len(skel3d):
            continue
        skel2d_view_set = draw_skel_reprojection(skel3d, proj_mat, views_path, triangulate_th, skel2d_view_set,
                                                 limb_map, v2)
    return skel2d_view_set


def draw_shelf_for_each_view(skel3d_set, proj_mat, views_path, type=0, view_set=None):
    from .skel_def import limb_map_SHELF14 as limb_map
    assert len(proj_mat) == len(views_path)
    if view_set is None:
        view_set = []
        for view_path_each in views_path:
            view_set.append(cv2.imread(view_path_each))
    assert len(view_set) == len(views_path)
    if type == 0:
        joint_color = (255, 0, 0)
        limb_color = (0, 255, 0)
    elif type == 1:
        joint_color = (0, 0, 255)
        limb_color = (255, 0, 0)
    else:
        joint_color = (255, 255, 255)
        limb_color = (255, 255, 255)
    for skel3d in skel3d_set:
        # make sure gt is not empty
        if not len(skel3d):
            continue
        # v2 feature
        if len(skel3d) == 2:
            skel3d = skel3d[0]
        # find valid joint map
        valid_joint_map = [np.sum(joint) != 0 for joint in skel3d]
        skel3d_data = np.ones((len(skel3d), 4))
        skel3d_data[:, :3] = np.array([item.squeeze() for item in skel3d])
        for idx, proj_each in enumerate(proj_mat):
            view_img = view_set[idx]

            # let's reproject the 3d joint back to 2d view
            joint2d = np.dot(proj_each, skel3d_data.T).T
            joint2d_normalized_x = joint2d[:, 0] / joint2d[:, 2]
            joint2d_normalized_y = joint2d[:, 1] / joint2d[:, 2]
            j2d_x_list, j2d_y_list = [], []
            for jIdx in range(len(joint2d_normalized_y)):
                j2d_x = int(joint2d_normalized_x[jIdx] + 0.5)
                j2d_y = int(joint2d_normalized_y[jIdx] + 0.5)
                j2d_x_list.append(j2d_x)
                j2d_y_list.append(j2d_y)
                if not valid_joint_map[jIdx]:
                    continue
                cv2.circle(view_img, (j2d_x, j2d_y), 2, joint_color, thickness=3, lineType=8, shift=0)
            # draw limb
            # override
            if len(valid_joint_map) == 19:
                limb_map = np.array(
                    [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                     [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [0, 3], [0, 4]]).T.tolist()
            if limb_map is not None and len(limb_map):
                for limbIdx in range(len(limb_map[0])):
                    if not valid_joint_map[limb_map[0][limbIdx]] or not valid_joint_map[limb_map[1][limbIdx]]:
                        continue
                    j2d_a = (j2d_x_list[limb_map[0][limbIdx]], j2d_y_list[limb_map[0][limbIdx]])
                    j2d_b = (j2d_x_list[limb_map[1][limbIdx]], j2d_y_list[limb_map[1][limbIdx]])
                    if sum(j2d_a) > 0 and sum(j2d_b) > 0:
                        cv2.line(view_img, j2d_a, j2d_b, limb_color, 1, 4)
    return view_set

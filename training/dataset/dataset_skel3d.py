import math
import random
import numpy as np
import torch.utils.data as data


def angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    return angle


def rot_normalization(skel, enable=False, pIdx=-1):
    if not enable:
        return skel
    hip_vec = skel[3] - skel[2]
    hip_vec = hip_vec[:2]
    ori_vec = np.array([1, 0])
    ang = angle(hip_vec, ori_vec)
    if pIdx == 2:
        ang += 45
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = math.cos(ang)
    rot_mat[0, 1] = -math.sin(ang)
    rot_mat[1, 0] = math.sin(ang)
    rot_mat[1, 1] = math.cos(ang)
    rot_mat[2, 2] = 1
    skel = np.dot(skel, rot_mat)
    return skel


def random_rotation(skel, ang):
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = math.cos(ang)
    rot_mat[0, 1] = -math.sin(ang)
    rot_mat[1, 0] = math.sin(ang)
    rot_mat[1, 1] = math.cos(ang)
    rot_mat[2, 2] = 1
    skel = np.dot(skel, rot_mat)
    return skel


class ShelfSkelDataset(data.Dataset):
    def __init__(self, raw_path, train=True, window_size=15,
                 mask_ratio_t=0.2, mask_ratio_j=0.2,
                 debug=False, eval=False,
                 rot=False, random_rot=False):
        if isinstance(raw_path, list):
            if train:
                self.raw_path = raw_path[0]
            else:
                self.raw_path = raw_path[1]
        else:
            self.raw_path = raw_path  # consist with old api

        self.rot = rot
        self.random_rot = random_rot
        self.train = train
        self.eval = eval
        self.debug = debug
        if train:
            self.eval = False
        self.limb_map = [[0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 2, 3, 8, 9],
                         [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 8, 9, 12, 12]]

        data = np.load(self.raw_path, allow_pickle=True).tolist()
        self.frame_index = np.array(data[-1])
        self.skel_num = len(self.frame_index)
        self.meta_data = data[:self.skel_num]
        self.data_length = np.sum(self.frame_index)
        if not train:
            for i in range(1, len(self.frame_index)):
                self.frame_index[i] += self.frame_index[i - 1]
            self.frame_index = np.insert(self.frame_index, 0, values=0, axis=0)
        self.joint_num = len(self.meta_data[0][0]["skel_data"])
        self.window_size = window_size
        self.mask_ratio_t = mask_ratio_t
        self.mask_ratio_j = mask_ratio_j
        self.num_masked_j = int(mask_ratio_j * self.joint_num + 0.5)
        self.num_masked_t = int(mask_ratio_t * self.window_size + 0.5)

        # strip meta data into list
        self.skel3d = []
        self.hip_center = []
        self.identity = []
        for pIdx, person in enumerate(self.meta_data):
            self.skel3d.append([rot_normalization(item["skel_data"], rot, pIdx) for item in person])
            self.hip_center.append([item["hip_center"] for item in person])
            self.identity.append([item["identity"] for item in person])

    def __getitem__(self, idx):
        frame_selected_bgn = -1
        frame_selected_end = -1
        # testing & evaluation
        if not self.train:
            person_selected = 0
            for i in range(1, len(self.frame_index)):
                frame_length = self.frame_index[i]
                frame_length_pre = self.frame_index[i - 1]
                if idx < frame_length:
                    if self.eval:
                        frame_selected = idx - frame_length_pre
                        frame_selected_bgn = np.clip(frame_selected - self.window_size, 0, None)
                        frame_selected_end = frame_selected + 1
                    else:
                        frame_selected = np.clip(idx, 0, frame_length - self.window_size)
                        frame_selected -= frame_length_pre
                        frame_selected_bgn = frame_selected
                        frame_selected_end = frame_selected + self.window_size
                    person_selected = i - 1
                    break
        # training
        else:
            # randomly select
            person_selected = random.randint(0, self.skel_num - 1)
            frame_selected = random.randint(0, self.frame_index[person_selected] - self.window_size - 1)
            frame_selected_bgn = frame_selected
            frame_selected_end = frame_selected + self.window_size
        # print stat
        if self.debug or self.eval:
            print("Picked p{} frame{} - frame{}".format(person_selected, frame_selected_bgn,
                                                        frame_selected_end - 1))
        # no mask
        if not self.eval:
            skel3d_selected = self.skel3d[person_selected][frame_selected_bgn:frame_selected_end]
            hip_center_selected = self.hip_center[person_selected][frame_selected_bgn:frame_selected_end]
        else:
            skel3d_selected = []
            hip_center_selected = []
            for i in range(frame_selected_bgn, frame_selected_end):
                skel3d_selected.append(self.skel3d[person_selected][i])
                hip_center_selected.append(self.hip_center[person_selected][i])
            for i in range(len(skel3d_selected), self.window_size):
                skel3d_selected.insert(0, skel3d_selected[0])
                hip_center_selected.insert(0, hip_center_selected[0])
            if len(skel3d_selected) != self.window_size:
                skel3d_selected = skel3d_selected[:self.window_size]
                hip_center_selected = hip_center_selected[:self.window_size]

        # V3T2.1
        skel3d_selected = np.vstack(skel3d_selected)
        # random rotation
        if self.random_rot:
            skel3d_selected = random_rotation(skel3d_selected, random.randint(0, 360))
        # frame-level masking
        rand_idx_t = np.random.rand(1, self.window_size).argsort(-1)
        mask_idx_t = rand_idx_t[:, :self.num_masked_t]
        unmask_idx_t = rand_idx_t[:, self.num_masked_t:]
        # joint-level masking
        mask_idx_j = []
        unmask_idx_j = []
        for i in range(self.window_size):
            if i in mask_idx_t:
                rand_idx_j = np.random.rand(1, self.joint_num).argsort(-1) + i * self.joint_num
                mask_idx_j.extend(rand_idx_j[:, :self.num_masked_j])
                unmask_idx_j.extend(rand_idx_j[:, self.num_masked_j:])
            else:
                fIdx = np.arange(self.joint_num).reshape(1, -1) + i * self.joint_num
                unmask_idx_j.extend(fIdx)
        # frame-level masking -> joint-level masking -> final masking (R_t * R_j * N_t * N_j)
        mask_idx = np.concatenate(mask_idx_j)
        unmask_idx = np.concatenate(unmask_idx_j)
        if self.debug:
            return skel3d_selected, (mask_idx, unmask_idx), hip_center_selected, person_selected
        elif self.eval:
            return skel3d_selected, (mask_idx, unmask_idx), person_selected, (frame_selected_bgn, frame_selected_end)
        elif self.train:
            return skel3d_selected, (mask_idx, unmask_idx)
        else:  # test
            return skel3d_selected, (mask_idx, unmask_idx)

    def __len__(self):
        return self.data_length


class ShelfHybridDataset(data.Dataset):
    def __init__(self, raw_path, train=True, eval=False, debug=False,
                 window_size=15, valid_joint_num=15, rot=False, random_rot=False,
                 mask_ratio_t=0.2, mask_ratio_j=0.2,
                 finetune=False, finetune_mode=-1):
        if isinstance(raw_path, list):
            if train:
                self.raw_path = raw_path[0]
            else:
                self.raw_path = raw_path[1]
        else:
            self.raw_path = raw_path  # consist with old api

        self.rot = rot
        self.random_rot = random_rot
        self.train = train
        self.eval = eval
        self.debug = debug
        if train:
            self.eval = False
        self.shelf_limb_map = [[0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 2, 3, 8, 9],
                               [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 8, 9, 12, 12]]
        self.openpose_limb_map = [[0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                  [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 20, 13, 14, 19, 17, 18]]
        self.openpose_to_shelf_map = [[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14],
                                      [13, 12, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5]]
        self.valid_joint_num = valid_joint_num

        data = np.load(self.raw_path, allow_pickle=True).tolist()
        self.skel_num = len(data) - 1
        self.meta_data = data[:self.skel_num]
        self.frame_index = np.zeros(self.skel_num, dtype=np.int64)
        # 14 + 21
        self.joint_num = len(self.meta_data[0][0]["skel_data"]) + len(self.meta_data[0][0]["openpose_skel"][0])
        self.window_size = window_size
        self.mask_ratio_t = mask_ratio_t
        self.mask_ratio_j = mask_ratio_j
        self.num_masked_j = int(mask_ratio_j * self.joint_num + 0.5)
        self.num_masked_t = int(mask_ratio_t * self.window_size + 0.5)
        self.finetune = finetune
        self.finetune_mode = finetune_mode
        # -> validation check
        assert self.finetune, "only used for fine-tune"
        assert self.finetune_mode >= 0, "invalid fine-tune mode: {}".format(self.finetune_mode)
        if self.finetune_mode == 0:
            assert self.num_masked_j == 14, "Fine-tune mode 0 mask ratio error: {}".format(self.num_masked_j)
        if self.finetune_mode == 1:
            assert self.num_masked_j == 21, "Fine-tune mode 0 mask ratio error: {}".format(self.num_masked_j)
        if self.finetune_mode == 2:
            assert self.num_masked_j > 14, "Fine-tune mode 2 mask ratio error: {}".format(self.num_masked_j)

        # strip meta data into list
        self.skel3d = [[] for i in range(self.skel_num)]
        self.hip_center = [[] for i in range(self.skel_num)]
        for pIdx, person in enumerate(self.meta_data):
            for item in person:
                shelf_data = item["skel_data"]
                shelf_hip_center = item["hip_center"]
                if not "openpose_skel" in item:
                    continue
                openpose_data = item["openpose_skel"][0]
                openpose_hip_center = item["openpose_skel"][1]
                # normalize shelf to openpose
                shelf_data = shelf_data + shelf_hip_center - openpose_hip_center

                # avoid insufficient data
                data_check = openpose_data.copy() + openpose_hip_center
                valid_joint_num = np.sum(np.all(data_check != 0., axis=1))
                # if valid_joint_num < self.valid_joint_num \
                # or np.all(openpose_hip_center == 0.) \
                # or np.all(data_check[15] == 0.) or np.all(data_check[16] == 0.):
                if valid_joint_num < self.valid_joint_num or np.all(openpose_hip_center == 0.):
                    continue
                self.skel3d[pIdx].append(np.vstack((shelf_data, openpose_data)))
                # robust check
                dist = np.average(np.sqrt(np.sum(np.power(shelf_hip_center - openpose_hip_center, 2))))
                if dist > 0.2:
                    continue

                assert dist < 0.2, "Invalid hip: {}-{}".format(pIdx, dist)
                self.hip_center[pIdx].append(np.vstack((shelf_hip_center, openpose_hip_center)))
                self.frame_index[pIdx] += 1

        self.data_length = np.sum(self.frame_index)
        if not train:
            for i in range(1, len(self.frame_index)):
                self.frame_index[i] += self.frame_index[i - 1]
            self.frame_index = np.insert(self.frame_index, 0, values=0, axis=0)

    def __getitem__(self, idx):
        frame_selected_bgn = -1
        frame_selected_end = -1
        # testing & evaluation
        if not self.train:
            person_selected = 0
            for i in range(1, len(self.frame_index)):
                frame_length = self.frame_index[i]
                frame_length_pre = self.frame_index[i - 1]
                if idx < frame_length:
                    if self.eval:
                        frame_selected = idx - frame_length_pre
                        frame_selected_bgn = np.clip(frame_selected - self.window_size, 0, None)
                        frame_selected_end = frame_selected + 1
                    else:
                        frame_selected = np.clip(idx, 0, frame_length - self.window_size)
                        frame_selected -= frame_length_pre
                        frame_selected_bgn = frame_selected
                        frame_selected_end = frame_selected + self.window_size
                    person_selected = i - 1
                    break
        # training
        else:
            # randomly select
            person_selected = random.randint(0, self.skel_num - 1)
            frame_selected = random.randint(0, self.frame_index[person_selected] - self.window_size - 1)
            frame_selected_bgn = frame_selected
            frame_selected_end = frame_selected + self.window_size
        # print stat
        if self.debug or self.eval:
            print("Picked p{} frame{} - frame{}".format(person_selected, frame_selected_bgn,
                                                        frame_selected_end - 1))

        # gather data
        skel3d_selected = []
        hip_center_selected = []
        if not self.eval:
            skel3d_selected = self.skel3d[person_selected][frame_selected_bgn:frame_selected_end]
            hip_center_selected = self.hip_center[person_selected][frame_selected_bgn:frame_selected_end]
        else:
            for i in range(frame_selected_bgn, frame_selected_end):
                skel3d_selected.append(self.skel3d[person_selected][i])
                hip_center_selected.append(self.hip_center[person_selected][i])
            for i in range(len(skel3d_selected), self.window_size):
                skel3d_selected.insert(0, skel3d_selected[0])
                hip_center_selected.insert(0, hip_center_selected[0])
            if len(skel3d_selected) != self.window_size:
                skel3d_selected = skel3d_selected[:self.window_size]
                hip_center_selected = hip_center_selected[:self.window_size]

        # V3T2.1
        skel3d_selected = np.vstack(skel3d_selected)
        # random rotation
        if self.random_rot:
            skel3d_selected = random_rotation(skel3d_selected, random.randint(0, 360))
        mask_idx, unmask_idx = self.fetch_mask_idx()

        if self.debug:
            return skel3d_selected, (mask_idx, unmask_idx), hip_center_selected, person_selected
        elif self.eval:
            return skel3d_selected, (mask_idx, unmask_idx), person_selected, (frame_selected_bgn, frame_selected_end)
        elif self.train:
            return skel3d_selected, (mask_idx, unmask_idx)
        else:  # test
            return skel3d_selected, (mask_idx, unmask_idx)

    def __len__(self):
        return self.data_length

    def convert_21_to_14(self, skel35):
        skel14 = skel35[:14]
        skel21 = skel35[14:]
        for idx in range(14):
            skel14[self.openpose_to_shelf_map[1][idx]] = skel21[self.openpose_to_shelf_map[0][idx]]
        return skel35

    def fetch_mask_idx(self):
        # frame-level masking
        rand_idx_t = np.random.rand(1, self.window_size).argsort(-1)
        mask_idx_t = rand_idx_t[:, :self.num_masked_t]
        unmask_idx_t = rand_idx_t[:, self.num_masked_t:]
        # joint-level masking
        mask_idx_j = []
        unmask_idx_j = []
        for i in range(self.window_size):
            if self.finetune:
                if self.finetune_mode == 2:
                    if i in mask_idx_t:
                        openpose_indices = np.random.rand(1, 21).argsort(-1) + 14 + i * self.joint_num
                        shelf_indices = np.arange(14).reshape(1, -1) + i * self.joint_num
                        rand_indices = np.hstack((shelf_indices, openpose_indices))
                        mask_idx_j.extend(rand_indices[:, :self.num_masked_j])
                        unmask_idx_j.extend(rand_indices[:, self.num_masked_j:])
                    else:
                        openpose_indices = np.arange(21).reshape(1, -1) + 14 + i * self.joint_num
                        shelf_indices = np.arange(14).reshape(1, -1) + i * self.joint_num
                        mask_idx_j.extend(shelf_indices)
                        unmask_idx_j.extend(openpose_indices)
        # frame-level masking -> joint-level masking -> final masking (R_t * R_j * N_t * N_j)
        mask_idx = np.concatenate(mask_idx_j)
        unmask_idx = np.concatenate(unmask_idx_j)
        return mask_idx, unmask_idx

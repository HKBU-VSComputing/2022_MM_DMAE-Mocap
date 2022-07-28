import os
import torch
import numpy as np

from util.skel_utils import util_convert_BODY21_to_SHELF14, util_convert_SHELF14_to_BODY21
from util.skel_utils import util_convert_MAE21_to_SHELF14, util_convert_MAE21_to_BODY21

from model.dmae import dmae_v5


def dmae_data_preprocess(skel3d_21, normalize=True, randomize=True, v2=False, v3=False):
    """
    Convert skeleton type from body21 to shelf14
    :param skel3d_21: source data
    :param normalize: flag for normalization
    :param randomize: DISCARDED
    :param v2: fine-tune, combined with body21 and shelf14
    :param v3: directly predict body21
    :return: target data, mid_hip, mask_ratio
    """
    # v2: shelf14 + openpose21
    if v2:
        skel3d_35 = np.zeros((14, 3))
        skel3d_21_ = np.array([item[1] for item in skel3d_21]).squeeze(axis=2)
        skel3d_35 = np.vstack((skel3d_35, skel3d_21_))
        mid_hip = skel3d_35[22].copy()
        mask = np.sum(skel3d_35, axis=1) == 0.
        missing_part_num = np.sum(mask)
        mask_ratio = missing_part_num / len(mask)
        mask_idx = np.where(mask)[0]
        unmask_idx = np.where(mask == False)[0]
        if normalize:
            skel3d_35[~mask, :] -= mid_hip
        if missing_part_num > 14:
            return skel3d_35, mid_hip, mask_ratio, (mask_idx, unmask_idx)
        else:
            return skel3d_35, mid_hip, 0, (mask_idx, unmask_idx)

    # v3: openpose21
    if v3:
        skel3d_21_ = np.array([item[1] for item in skel3d_21]).squeeze(axis=2)
        mid_hip = skel3d_21_[8].copy()
        mask = np.sum(skel3d_21_, axis=1) == 0.
        missing_part_num = np.sum(mask)
        mask_ratio = missing_part_num / len(mask)
        mask_idx = np.where(mask)[0]
        unmask_idx = np.where(mask == False)[0]
        if normalize:
            skel3d_21_[~mask, :] -= mid_hip
        return skel3d_21_, mid_hip, mask_ratio, (mask_idx, unmask_idx)

    skel3d_14 = np.array(util_convert_BODY21_to_SHELF14(skel3d_21))
    skel3d_14 = skel3d_14.squeeze(2)
    mid_hip = skel3d_14[-1].copy()
    skel3d_14 = skel3d_14[:-1, :]  # discard mid-hip

    mask = np.sum(skel3d_14, axis=1) == 0.
    # note: actually, 13(TOP HEAD) is missing, 12(BOTTOM HEAD)'s value is also missing
    if mask[13] or mask[12]:
        mask[12] = True
        mask[13] = True
        skel3d_14[12][:] = 0.
        skel3d_14[13][:] = 0.
    missing_part_num = np.sum(mask)
    mask_ratio = missing_part_num / len(mask)
    mask_idx = np.where(mask)[0]
    unmask_idx = np.where(mask == False)[0]

    if normalize:
        skel3d_14[~mask, :] -= mid_hip

    return skel3d_14, mid_hip, mask_ratio, (mask_idx, unmask_idx)


def dmae_model_prepare(device,
                       model_path=None,
                       patch_num=525, patch_dim=3, mask_ratio=0.4, frame_num=15,
                       seed=0, ft=True):
    if not os.path.exists(model_path):
        print("No file found at {}".format(model_path))
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = dmae_v5(patch_num, patch_dim, frame_num, mask_ratio, ft)
    client_states = torch.load(model_path)
    state_dict = client_states['model']
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


def dmae_model_process(model, data, mask_data, device, dim=15):
    data_tensor = torch.from_numpy(data).to(device).unsqueeze(0).to(torch.float32)
    mask_idx, unmask_idx = mask_data[0], mask_data[1]
    mask_idx_tensor = torch.from_numpy(mask_idx).to(device).unsqueeze(0)
    unmask_idx_tensor = torch.from_numpy(unmask_idx).to(device).unsqueeze(0)

    with torch.no_grad():
        outputs, labels, mask_idx, loss_ = model(data_tensor, (mask_idx_tensor, unmask_idx_tensor))
        outputs = outputs.cpu().numpy()[0]
        mask_idx = mask_idx.cpu().numpy()[0]

        skel_pred_full = data.copy()
        skel_pred_full[mask_idx] = outputs
        # store the last frame
        skel_pred_full = skel_pred_full.reshape(dim, -1, 3)
        skel_pred_full = skel_pred_full[-1]

    return skel_pred_full


def dmae_data_postprocess(skel_pred, skel_src, normalizer, v2=False, v3=False):
    # v2: shelf14 + openpose21
    if v2:
        body21 = util_convert_MAE21_to_BODY21(skel_pred, skel_src, normalizer)
        shelf14_pred = util_convert_MAE21_to_SHELF14(skel_pred, skel_src, normalizer)
        return (body21, shelf14_pred)

    # v3: openpose21
    if v3:
        return util_convert_MAE21_to_BODY21(skel_pred, skel_src, normalizer, v2=False)

    return util_convert_SHELF14_to_BODY21(skel_pred, skel_src, normalizer)


class DMAEQueueController:
    def __init__(self, length=15, stride=3, ch=14, debug=False):
        self.skel_num = 0
        self.length = length
        self.stride = stride
        self.queue = []
        self.th = 0.3
        self.ch = ch
        self.debug = debug

    def search(self, data):
        # calc dist
        dist_set = []
        for i in range(self.skel_num):
            skel_a = np.sum(data)
            skel_b = np.sum(self.queue[i][-1][:2])
            mask = np.logical_and(np.sum(skel_a, axis=1) != 0, np.sum(skel_b, axis=1) != 0)
            dist = np.average(np.sqrt(np.sum(np.power(skel_a[mask] - skel_b[mask], 2), axis=1)))
            dist_set.append(dist)
        return dist_set

    def append(self, skel, hip=None, mask_data=None):
        if hip is None:
            hip = np.zeros(3)
        assert mask_data is not None
        # init
        if self.skel_num == 0:
            self.queue.append([(skel, hip, mask_data) for i in range(self.length)])
            self.skel_num += 1
        else:
            dist_set = self.search((skel, hip))
            # find the closest skel
            min_dist_idx = np.argmin(dist_set)
            if dist_set[min_dist_idx] < self.th:
                # push and pop
                self.queue[min_dist_idx].pop(0)
                self.queue[min_dist_idx].append((skel, hip, mask_data))
            # this is a new skel
            else:
                self.queue.append([(skel, hip, mask_data) for i in range(self.length)])
                self.skel_num += 1
                if self.debug:
                    print("New skel!!!")

    def fetch(self, skel, hip=None):
        dist_set = self.search((skel, hip))
        # find the closest skel
        min_dist_idx = np.argmin(dist_set)
        if self.debug:
            print("Fetch at {}".format(min_dist_idx))
        if min_dist_idx > self.skel_num:
            return None
        out_skel = np.array([data[0] for data in self.queue[min_dist_idx]]).reshape(-1, 3)
        out_hip = np.array([data[1] for data in self.queue[min_dist_idx]]).reshape(1, -1)
        mask_idx = [data[2][0] + fIdx * self.ch for fIdx, data in enumerate(self.queue[min_dist_idx])]
        unmask_idx = [data[2][1] + fIdx * self.ch for fIdx, data in enumerate(self.queue[min_dist_idx])]
        mask_idx = np.concatenate(mask_idx)
        unmask_idx = np.concatenate(unmask_idx)
        return out_skel, out_hip, (mask_idx, unmask_idx)

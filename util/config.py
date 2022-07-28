import argparse


class BaseCfg:
    def __init__(self, args):
        # adaptive triangulation
        self.min_view_skel = 3
        self.min_view_joint = 3
        self.filter_prob = 0.2
        self.min_joint_num = 20
        # dmae
        self.dmae_enable = not args.no_dmae
        self.queue_ch = 35
        # ablation study
        self.cut_data = False
        self.cut_camera_choose = [1, 2, 3, 4]
        # snapshot
        self.snapshot_flag = args.snapshot
        self.snapshot_rescale_ratio = 0.7
        # output
        self.out_folder = "output"

    def __repr__(self):
        return "ATM: {}-{}-{}; Out: {}; DMAE: {}; snapshot: {}".format(self.min_view_skel, self.min_view_joint,
                                                                       self.filter_prob, self.out_folder,
                                                                       self.dmae_enable, self.snapshot_flag)


class ShelfCfg(BaseCfg):
    def __init__(self, args):
        super().__init__(args)
        self.data_root = "data/shelf"
        self.img_root = "data/shelf/sequences"
        if self.dmae_enable:
            self.out_folder = "data/shelf/output/npy"
            self.snp_folder = "data/shelf/output/snapshot"
        else:
            self.out_folder = "data/shelf/output_no_dmae/npy"
            self.snp_folder = "data/shelf/output_no_dmae/snapshot"

        self.camera_param_path = "data/shelf/camera_params.npy"
        self.skel2d_dict_path = "data/shelf/shelf_eval_2d_detection_dict.npy"
        self.model_path = "data/shelf/checkpoint-best.pth"

        self.img_suffix = ".png"
        self.view_list = ["img_0", "img_1", "img_2", "img_3", "img_4"]
        self.frame_start = 0
        self.frame_end = 301


class ShelfEvaCfg(ShelfCfg):
    def __init__(self, args):
        super().__init__(args)
        self.gt_path = "data/shelf/shelf_3d_eval_4dassoc.npy"
        self.eval_snapshot_flag = True
        if self.dmae_enable:
            self.out_folder = "data/shelf/output"
            self.pred_root = "data/shelf/output/npy"
            self.eval_snp_folder = "data/shelf/output/eval_snapshot"
        else:
            self.out_folder = "data/shelf/output_no_dmae"
            self.pred_root = "data/shelf/output_no_dmae/npy"
            self.eval_snp_folder = "data/shelf/output_no_dmae/eval_snapshot"

        self.eval_pck_th = 0.2
        self.eval_reid_th = 0.5

    def __repr__(self):
        return "In: {}; Out: {}; DMAE: {}; snapshot: {}".format(self.pred_root, self.out_folder, self.dmae_enable,
                                                                self.eval_snapshot_flag)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple parser for D-MAE Mocap framework")
    parser.add_argument('--no-dmae', action='store_true', default=False, help='Disable D-MAE function.')
    parser.add_argument('--snapshot', action='store_true', default=False, help='Store snapshot for visualization.')
    args = parser.parse_args()
    return args

import sys
import cv2
import os
import os.path
import numpy as np

# Import OpenPose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    # Change me!!! ###########################################################
    sys.path.append(dir_path + '/../bin/python/openpose/Release')
    ##########################################################################

    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.'
          'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
# Change me!!! ###########################################################
params["model_folder"] = "../models/"
##########################################################################

"""
0 for original res
1 for network output size
2 for output size
3 for [0 (top-left), 1(bottom-right)]
4 for [-1, 1]
"""
params["keypoint_scale"] = 3

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def filter_joint(skel, filter_list):
    skel_filtered = []
    for idx, joint in enumerate(skel):
        if idx in filter_list:
            continue
        skel_filtered.append(joint)
    return skel_filtered


def extract_skeletons_dict_per_frame(src_path, dst_path, snapshot=True, np_save=True, resize=True):
    """
    Extract 2D skeletons for each view, each frame
    :param src_path: the input RGB image path
    :param dst_path: the output 2D detection result
    :param snapshot: flag for snapshot or not
    :param np_save: flag for saving detection results in numpy format or not
    :param resize: flag for resizing the snapshot or not
    :return: 2D detection result of the input image
    """
    # parse filename
    npy_filename = os.path.basename(src_path).split(".")[0] + '.npy'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    out_npy_path = os.path.join(dst_path, npy_filename)
    out_img_path = os.path.join(dst_path, os.path.basename(src_path)[:-3] + 'jpg')

    # jump processed file
    if os.path.exists(out_npy_path):
        print("{} jumped".format(out_npy_path))
        return None

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(src_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    # Fetch Output
    # 1. snapshot
    if snapshot:
        snapshot_img = datum.cvOutputData
        x, y = snapshot_img.shape[0:2]
        if min(x, y) > 360:
            y = int(y / 4)
            x = int(x / 4)
        if resize:
            snapshot_img = cv2.resize(snapshot_img, (y, x))
        cv2.imwrite(out_img_path, snapshot_img)

    if datum.poseKeypoints is None:
        return None
    keypoints = datum.poseKeypoints.copy()
    identity, joint_size, ch = keypoints.shape
    print("{} has {} skeletons with {} joints".format(src_path, identity, joint_size))
    # filter joint: r/l-little toe/heel
    filter_list = [20, 21, 23, 24]
    skel_data = []
    for id, person in enumerate(keypoints):
        filtered_skel = filter_joint(person, filter_list)
        filtered_skel = np.array(filtered_skel)
        skel = {
            "identity": id,
            "meta": person,
            "joint_define": {
                0: "Nose",
                1: "Neck",
                2: "RShoulder",
                3: "RElbow",
                4: "RWrist",
                5: "LShoulder",
                6: "LElbow",
                7: "LWrist",
                8: "MidHip",
                9: "RHip",
                10: "RKnee",
                11: "RAnkle",
                12: "LHip",
                13: "LKnee",
                14: "LAnkle",
                15: "REye",
                16: "LEye",
                17: "REar",
                18: "LEar",
                19: "LBigToe",
                20: "LSmallToe",
                21: "LHeel",
                22: "RBigToe",
                23: "RSmallToe",
                24: "RHeel",
                25: "Background"
            },
            "limb_define": [[0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16,
                             19, 22],
                            [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 22, 24, 13, 14, 19, 21, 17,
                             18, 20, 23]],
            "filter_list": filter_list,
            "filtered_skel": filtered_skel
        }
        skel_data.append(skel)
    if np_save:
        np.save(out_npy_path, skel_data)
    return skel_data


"""
Used for extracting data in Triangulation format
"""
if __name__ == '__main__':

    # Change me!!! ###########################################################
    img_dir = "\Shelf\Evaluate"
    dat_dir = "\shelf_data"
    ##########################################################################

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if 'jpg' in file or 'png' in file:
                img_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dat_dir, root.split("\\")[-1])
                extract_skeletons_dict_per_frame(img_file_path, dst_file_path, resize=False, snapshot=False)

    # arrange data and store
    view_data = {}
    for view_root in os.listdir(dat_dir):
        view_data[view_root] = {}
        for frame in os.listdir(os.path.join(dat_dir, view_root)):
            if "npy" in frame:
                frame_name = frame.split(".")[0]
                skel_data = np.load(os.path.join(os.path.join(dat_dir, view_root), frame), allow_pickle=True).tolist()
                view_data[view_root][frame_name] = skel_data
    assert len(view_data) == 5, "Wrong view number for Shelf at folder {}".format(dat_dir)
    for view_name, view in view_data.items():
        assert len(view) == 301, "Wrong frame number for Shelf at view {}".format(view_name)
    np.save(os.path.join(dat_dir, "shelf_eval_2d_detection_dict.npy"), view_data)
    print("=> Done!")

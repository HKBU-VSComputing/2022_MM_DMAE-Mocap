import os
import numpy as np

# NOTE!!!
# R[0:2, :] = -R[0:2, :]
# T[0:2] = -T[0:2]
# according to the original R and T
# use this transform to meet the original P value
camera_0 = {
    "P": [
        [1080.661255, -474.447113, 2.303060, 711.040283],
        [245.451584, 176.193542, -1086.353027, 1960.634033],
        [0.758863, 0.649664, -0.045434, 2.624511],
    ],
    "K": [
        [1063.512085, 0.000000, 511.738251],
        [0.000000, 1071.863647, 350.088287],
        [0.000000, 0.000000, 1.000000],
    ],
    "R": [
        [0.650977, -0.758717, 0.024027],
        [-0.018862, -0.047810, -0.998678],
        [0.758863, 0.649664, -0.045434],
    ],
    "T": [
        -0.594278, 0.971974, 2.624511
    ],
    "res": [
        1032.000000, 776.000000
    ],
    "id": 0
}

camera_1 = {
    "P": [
        [502.943146, -1106.368042, -7.693093, 2118.563965],
        [344.384766, -13.377553, -1097.217651, 2272.101318],
        [0.999426, -0.016967, -0.029322, 3.543943],
    ],
    "K": [
        [1097.697754, 0.000000, 521.652161],
        [0.000000, 1086.668457, 376.587067],
        [0.000000, 0.000000, 1.000000],
    ],
    "R": [
        [-0.016771, -0.999835, 0.006926],
        [-0.029435, -0.006431, -0.999546],
        [0.999426, -0.016967, -0.029322],
    ],
    "T": [
        0.245840, 0.862727, 3.543943
    ],
    "res": [
        1032.000000, 776.000000
    ],
    "id": 1
}

camera_2 = {
    "P": [
        [-615.764343, -1076.782959, -244.496887, 2524.561523],
        [-228.749222, 191.121994, -1135.570679, 2241.660645],
        [0.488586, -0.682410, -0.543691, 3.893979],
    ],
    "K": [
        [1130.065552, 0.000000, 566.884338],
        [0.000000, 1112.470337, 375.212708],
        [0.000000, 0.000000, 1.000000],
    ],
    "R": [
        [-0.789986, -0.610527, 0.056380],
        [-0.370413, 0.401962, -0.837389],
        [0.488586, -0.682410, -0.543691],
    ],
    "T": [
        0.280626, 0.701673, 3.893979
    ],
    "res": [
        1032.000000, 776.000000
    ],
    "id": 2
}

camera_3 = {
    "P": [
        [-1146.696289, -276.914490, -170.500946, 2291.196777],
        [16.893785, -166.067093, -1117.838623, 2139.395752],
        [-0.220150, -0.951779, -0.213663, 3.760249],
    ],
    "K": [
        [1056.162598, 0.000000, 552.435730],
        [0.000000, 1059.639648, 393.180389],
        [0.000000, 0.000000, 1.000000],
    ],
    "R": [
        [-0.970568, 0.235647, -0.049676],
        [0.097630, 0.196438, -0.975644],
        [-0.220150, -0.951779, -0.213663],
    ],
    "T": [
        0.202527, 0.623740, 3.760249
    ],
    "res": [
        1032.000000, 776.000000
    ],
    "id": 3
}

camera_4 = {
    "P": [
        [-686.368408, 971.058655, -147.044647, 2576.208496],
        [-90.658463, -51.368141, -1134.439575, 1994.441162],
        [-0.952896, -0.195466, -0.231908, 4.038814],
    ],
    "K": [
        [1089.654175, 0.000000, 498.329620],
        [0.000000, 1080.999390, 359.514832],
        [0.000000, 0.000000, 1.000000],
    ],
    "R": [
        [-0.194109, 0.980554, -0.028888],
        [0.233045, 0.017488, -0.972309],
        [-0.952896, -0.195466, -0.231908],
    ],
    "T": [
        0.517180, 0.501783, 4.038814
    ],
    "res": [
        1032.000000, 776.000000
    ],
    "id": 4
}


def get_eval_data_4dassoc(src_path, dst_np_path):
    if not os.path.exists(dst_np_path):
        with open(src_path, "r") as f:
            line = f.readlines()
            line_num = len(line)
            # read header
            skel_type, frame_num = line[0].split()
            skel_type, frame_num = int(skel_type), int(frame_num)
            eval_data = [[[] for p in range(4)] for i in range(frame_num)]

            i = 1
            frame_idx = 0
            while i < line_num:
                skel_num = int(line[i])
                for j in range(skel_num):
                    skel_id = int(line[i + 1])
                    skel_data = np.array([line[i + 2].split(),
                                          line[i + 3].split(),
                                          line[i + 4].split(),
                                          line[i + 5].split()],
                                         dtype=np.float64)
                    eval_data[frame_idx][skel_id] = skel_data[:3, :].T
                    i += 5
                i += 1
                frame_idx += 1
            np.save(dst_np_path, np.array(eval_data))
    print("=> Done!")


def get_shelf_camera_params(camera_np_path):
    camera_set = [camera_0, camera_1, camera_2, camera_3, camera_4]
    if not os.path.exists(camera_np_path):
        for camera in camera_set:
            R = np.array(camera["R"])
            T = np.array(camera["T"]).T.reshape(3, -1)
            K = np.array(camera["K"])
            RT = np.hstack((R, T))
            P_comp = np.dot(K, RT)
            Pos = -np.dot(R.T, T)
            camera["Pos"] = Pos
            # checkout
            print("====")
            print(P_comp)
            print(RT)
            print(Pos)
        np.save(camera_np_path, camera_set)

    if os.path.exists(camera_np_path):
        camera_set = np.load(camera_np_path, allow_pickle=True).tolist()
        print(camera_set)


if __name__ == '__main__':
    # camera params
    camera_path = "data/shelf/camera_params.npy"
    get_shelf_camera_params(camera_path)

    # ground-truth
    src_path = r"D:\workspace\code\4d_association\data\shelf\gt.txt"
    dst_path = r"D:\workspace\code\MAE-pytorch\openpose\shelf_3d_eval_4dassoc.npy"
    get_eval_data_4dassoc(src_path, dst_path)

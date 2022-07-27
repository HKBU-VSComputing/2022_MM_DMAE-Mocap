# Skeleton Definition
# BODY25
limb_map_BODY25 = [[0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 19, 22],
                   [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 22, 24, 13, 14, 19, 21, 17, 18, 20, 23]]
# BODY21 (filter with J23(RSmallToe), J24(RHeel), J20(LSmallToe), J21(LHeel))
limb_map_BODY21 = [[0, 0, 0, 1, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                   [1, 15, 16, 2, 5, 3, 9, 4, 6, 12, 7, 9, 12, 10, 11, 20, 13, 14, 19, 17, 18]]
# SHELF14
limb_map_SHELF14 = [[0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 2, 3, 8, 9],
                    [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 8, 9, 12, 12]]
eval_map_SHELF14 = [[9, 7, 10, 6, 3, 1, 4, 0, 12, 12],
                    [10, 8, 11, 7, 4, 2, 5, 1, 13, 14]]
eval_map_SKEL17 = [[6, 7, 8, 7, 12, 11, 16, 15, 0, 17],
                   [8, 5, 10, 9, 14, 13, 14, 13, 17, 18]]
eval_map_SKEL19 = [[6, 11, 12, 11, 3, 2, 8, 7, 4, 1],
                   [12, 5, 16, 15, 8, 7, 14, 13, 1, 0]]
eval_map_SHELF14_name = ["Left Upper Arm", "Right Upper Arm", "Left Lower Arm", "Right Lower Arm",
                         "Left Upper Leg", "Right Upper Leg", "Left Lower Leg", "Right Lower Leg",
                         "Head", "Torso"]

# Convert skel from Type A to Type B
# from BODY21 to SHELF14
# src: Nose,     Neck
# dst: TOP_HEAD, BOTTOM_HEAD
# SHELF14 got extra joint: Mid_Hip(14)
convert_map_BODY21_SHELF14 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                              [13, 12, 8, 7, 6, 9, 10, 11, 14, 2, 1, 0, 3, 4, 5, -1, -1, -1, -1, -1, -1]]
convert_map_BODY21_SHELF14_v2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                 [13, 12, 8, 7, 6, 9, 10, 11, 14, 2, 1, 0, 3, 4, 5]]
convert_map_BODY25_SKEL19 = [4, 1, 5, 11, 15, 6, 12, 16, 0, 2, 7, 13, 3, 8, 14, -1, -1, 9, 10, 17, -1, -1, 18, -1, -1]
convert_map_BODY25_SKEL17 = [0, -1, 6, 8, 10, 5, 7, 9, -1, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, -1, -1, -1, -1, -1, -1]
convert_map_PRED21_SKEL19 = [4, 1, 5, 11, 15, 6, 12, 16, 0, 2, 7, 13, 3, 8, 14, -1, -1, 9, 10, 17, 18]
convert_map_SKEL19_SHELF14 = [14, 12, 2, 3, 13, 8, 9, 1, 4, -1, -1, 7, 10, 0, 5, 6, 11, -1, -1]

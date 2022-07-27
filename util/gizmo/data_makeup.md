# Prepare the data

This is the instruction for how to prepare your own 2D detection data. As mentioned in the paper, we adopt [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), a bottom-up 2D HPE detector to detect 2D joints for each view due to its efficiency and accuracy trade-off. Theoretically, you can use any other detectors you want, especially some more powerful ones such as [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation).


## Prepare Shelf data

Run ``python shelf_makeup.py`` to convert Shelf's camera parameters and ground-truth for the evaluation sequence into ``numpy`` files.

## Prepare 2D skeleton data

We provide a simple python script ``openpose_wrapper.py`` here to detect 2D skeletons. Please modify ``Line 12`` and ``Line 25`` to configure the OpenPose. In ``Line 152``, the input image directory is ``img_dir``. And in ``Line 153``, the output directory of detection results is ``dat_dir``.

We don't check the validity, such as the view name, the sequence length, the image name and so on. But please make sure that in each view's sub-folder, the image name should be consistent. The file structure of the input image directory should be:

```
IMG_DIR/
    └── view_0/
        └── 0000.png
        └── 0001.png
        └── ...
        └── xxxx.png
    └── view_1/
        └── 0000.png
        └── 0001.png
        └── ...
        └── xxxx.png
    └── ...
    └── view_x/
        └── 0000.png
        └── 0001.png
        └── ...
        └── xxxx.png
```

The output folders will be created automatically and the structure will be the same as the input.
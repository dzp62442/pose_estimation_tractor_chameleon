# %%
import os
import numpy as np
import cv2 as cv
import glob
from pathlib import Path
print(cv.__version__)

img_type = 'jpg'
calib_dir = "./my_imgs/pic3/calib_imgs"
rotate_dir = "./my_imgs/pic3/rotate_imgs/"

grid_size = 0.03  # 每个网格尺寸，单位米
grid_shape = (8, 6)  # 内角点列数、行数

# %%
# 相机标定
def calib(grid_shape, grid_size, img_dir, img_type):
    print(grid_shape)  # cv::Size(columns，rows)
    w, h = grid_shape

    # cp_int: 世界坐标系中int格式的角点序号坐标，如 (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,0:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # cp_world: 世界坐标系中的角点坐标
    cp_world = cp_int * grid_size

    obj_points = []  # 空间坐标系中的点
    img_points = []  # 像素坐标系中的点
    images = glob.glob(img_dir + os.sep + '**.' + img_type)  # 查找符合规则的文件路径名

    draw_save_dir = img_dir.replace('calib_imgs', 'calib_corners')
    if not Path(draw_save_dir).is_dir():
        os.mkdir(draw_save_dir)

    for i, fname in enumerate(images):
        img = cv.imread(fname)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 查找角点，cp_img: 像素坐标系中的角点坐标
        ret, cp_img = cv.findChessboardCorners(gray_img, grid_shape, None)
        if ret == True:
            obj_points.append(cp_world)
            img_points.append(cp_img)
            draw_img = cv.drawChessboardCorners(img, grid_shape, cp_img, ret)
            cv.imwrite(os.path.join(draw_save_dir, str(i)+'.jpg'), draw_img)
        else:
            print(fname)
            raise RuntimeError("Find Chessboard Corners Error !")

    # 相机标定
    ret, K, coff_dis, v_rot, v_trans = cv.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None, flags=(cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K3))
    print("Ret:", ret)
    print("Internal matrix:\n", K)
    print("Distortion Cofficients:\n", coff_dis)
    # print("Rotation vectors:\n", v_rot[0])
    # print("Translation vectors:\n", v_trans[0])
    # 计算重投影误差
    total_error = 0
    for i in range(len(obj_points)):
        # 世界坐标系中的点重投影到像素坐标系中
        img_points_repro, _ = cv.projectPoints(obj_points[i], v_rot[i], v_trans[i], K, coff_dis)
        error = cv.norm(img_points[i], img_points_repro, cv.NORM_L2) / len(img_points_repro)
        total_error += error
    print("Average Error of Reproject:", total_error/len(obj_points))

    return K, coff_dis

K, coff_dis = calib(grid_shape, grid_size, calib_dir, img_type);

# %%
# PnP问题求解相机位姿
def usePnP(idx, K, coff_dis, grid_shape, grid_size, img_dir, img_type):
    w, h = grid_shape

    draw_save_dir = img_dir.replace('rotate_imgs', 'rotate_corners')
    if not Path(draw_save_dir).is_dir():
        os.mkdir(draw_save_dir)

    # cp_int: 世界坐标系中int格式的角点序号坐标，如 (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,0:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # cp_world: 世界坐标系中的角点坐标
    cp_world = cp_int * grid_size

    # cp_img: 像素坐标系中的角点坐标
    images = glob.glob(img_dir + os.sep + '**.' + img_type)  # 查找符合规则的文件路径名
    img = cv.imread(images[idx])
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, cp_img = cv.findChessboardCorners(gray_img, grid_shape, None)
    if ret == True:
        draw_img = cv.drawChessboardCorners(img, grid_shape, cp_img, ret)
        cv.imwrite(os.path.join(draw_save_dir, str(idx)+'.jpg'), draw_img)
    else:
        print(images[idx])
        raise RuntimeError("Find Chessboard Corners Error !")

    # 估计相机位姿
    _, v_rot, t = cv.solvePnP(cp_world, cp_img, K, coff_dis)
    # print("Rotation vectors:\n", v_rot)
    # print("Translation vectors:\n", t)

    # 计算累积重投影误差
    total_error = 0
    # 世界坐标系中的点重投影到像素坐标系中
    cp_repro, _ = cv.projectPoints(cp_world, v_rot, t, K, coff_dis)
    for i in range(len(cp_world)):
        error = cv.norm(cp_repro[i], cp_img[i], cv.NORM_L2) ** 2
        total_error += error
    total_error /= len(cp_world)
    print("Average Error of Reproject:", total_error)

    return v_rot, t

# %%
# 使用PnP求解世界坐标系到相机坐标系的变换矩阵
pic_num = 9  # 图像数目
vr_w2i = []  # 世界坐标系到相机坐标系i的旋转向量
t_w2i = []  # 世界坐标系到相机坐标系i的平移向量
R_w2i = []  # 世界坐标系到相机坐标系i的旋转矩阵

for i in range(pic_num):
    vr, t = usePnP(i, K, coff_dis, grid_shape, grid_size, rotate_dir, img_type)
    R = cv.Rodrigues(vr)[0]
    vr_w2i.append(vr)
    t_w2i.append(t)
    R_w2i.append(R)

# %%
# 计算第i帧到第i+1帧之间的位姿变换，求旋转向量表示
R_i2i1 = []  # 第i帧到第i+1帧的相机旋转矩阵
vr_i2i1 = []  # 第i帧到第i+1帧的相机旋转向量

for i in range(pic_num-1):
    R = R_w2i[i+1] @ R_w2i[i].T
    vr = cv.Rodrigues(R)[0]
    print(vr / np.linalg.norm(vr), np.linalg.norm(vr) * 180 / np.pi)
    R_i2i1.append(R)
    vr_i2i1.append(vr)

# %%
600 * 0.0514

# %%
# 计算第0帧到第i帧之间的位姿变换，求旋转向量表示
R_02i = []  # 第0帧到第i帧的相机旋转矩阵
vr_02i = []  # 第0帧到第i帧的相机旋转向量

for i in range(1, pic_num):
    R = R_w2i[i] @ R_w2i[0].T
    vr = cv.Rodrigues(R)[0]
    print(vr / np.linalg.norm(vr), np.linalg.norm(vr) * 180 / np.pi)
    R_02i.append(R)
    vr_02i.append(vr)

# %%
# 计算第i帧到第0帧之间的位姿变换，求旋转向量表示
R_i20 = []  # 第i帧到第0帧的相机旋转矩阵
vr_i20 = []  # 第i帧到第0帧的相机旋转向量

for i in range(1, pic_num):
    R = R_w2i[0] @ R_w2i[i].T
    vr = cv.Rodrigues(R)[0]
    print(vr / np.linalg.norm(vr), np.linalg.norm(vr) * 180 / np.pi)
    R_i20.append(R)
    vr_i20.append(vr)



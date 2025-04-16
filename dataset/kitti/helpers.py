import os
import numpy as np
# import scenerf.data.utils.fusion as fusion
# import open3d as o3d
from PIL import Image


# def make_open3d_point_cloud(xyz, color=None):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     if color is not None:
#         if len(color) != len(xyz):
#             color = np.tile(color, (len(xyz), 1))
#         pcd.colors = o3d.utility.Vector3dVector(color)
#     return pcd
def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot

def kitti_string_to_float(str):
    return float(str.split("e")[0]) * 10 ** int(str.split("e")[1])

def apply_transform(pts, T):
    """
    cam_ptx_from: B, 3
    """
    ones = np.ones((pts.shape[0], 1))
    homo_pts = np.concatenate([pts, ones], axis=1)
    homo_cam_pts_to = (T @ homo_pts.T).T
    cam_pts_to = homo_cam_pts_to[:, :3]

    return cam_pts_to


def dump_xyz(P):
    return P[0:3, 3]


def read_rgb(path):
    img = Image.open(path).convert("RGB")

    # PIL to numpy
    img = np.array(img, dtype=np.float32, copy=False) / 255.0
    img = img[:370, :1220, :]  # crop image        

    return img


def read_poses(path):
    # Read and parse the poses
    poses = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            poses.append(T_w_cam0)
    return poses
    

def read_calib(calib_path):
    """
    Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = calib_all["P2"].reshape(3, 4)
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)

    T2 = np.eye(4)
    T2[0, 3] = calib_out["P2"][0, 3] / calib_out["P2"][0, 0]
    calib_out["T_cam0_2_cam2"] = T2
    return calib_out

def get_data_path(dataset_type, seq, name):
    if dataset_type == 'odometry':
        data_path = f"dataset/sequences/{seq}/{name}"
    elif dataset_type == 'mot':
        data_path = f'{name}/{seq}'
    elif dataset_type =='kitti360':
        sequence_folder = f'2013_05_28_drive_{seq}_sync'
        if name == 'image_2':
            data_path = f'data_2d_raw/{sequence_folder}/image_00/data_rect'
        if name == 'semantic':
            data_path = f'semantic/{seq}'
        elif name == 'velodyne':
            data_path = f'data_3d_raw/{sequence_folder}/velodyne_points/data'
        
    return data_path
    
def get_poses_kittimot(basedir, oxts_path_tracking=None, selected_frames=None):
    """
    Extract poses and calibration information from the KITTI dataset.

    This function processes the OXTS data (GPS/IMU) and extracts the
    pose information (translation and rotation) for each frame. It also
    retrieves the calibration information (transformation matrices and focal length)
    required for further processing.

    Args:
        basedir (str): The base directory containing the KITTI dataset.
        oxts_path_tracking (str, optional): Path to the OXTS data file for tracking sequences.
            If not provided, the function will look for OXTS data in the basedir.
        selected_frames (list, optional): A list of frame indices to process.
            If not provided, all frames in the dataset will be processed.

    Returns:
        tuple: A tuple containing the following elements:
            poses (np.array): An array of 4x4 pose matrices representing the vehicle's
                position and orientation for each frame (IMU pose).
            calibrations (dict): A dictionary containing the transformation matrices
                and focal length obtained from the calibration files.
            focal (float): The focal length of the left camera.
    """

    def oxts_to_pose(oxts):
        """
        OXTS (Oxford Technical Solutions) data typically refers to the data generated by an Inertial and GPS Navigation System (INS/GPS) that is used to provide accurate position, orientation, and velocity information for a moving platform, such as a vehicle. In the context of the KITTI dataset, OXTS data is used to provide the ground truth for the vehicle's trajectory and 6 degrees of freedom (6-DoF) motion, which is essential for evaluating and benchmarking various computer vision and robotics algorithms, such as visual odometry, SLAM, and object detection.

        The OXTS data contains several important measurements:

        1. Latitude, longitude, and altitude: These are the global coordinates of the moving platform.
        2. Roll, pitch, and yaw (heading): These are the orientation angles of the platform, usually given in Euler angles.
        3. Velocity (north, east, and down): These are the linear velocities of the platform in the local navigation frame.
        4. Accelerations (ax, ay, az): These are the linear accelerations in the platform's body frame.
        5. Angular rates (wx, wy, wz): These are the angular rates (also known as angular velocities) of the platform in its body frame.

        In the KITTI dataset, the OXTS data is stored as plain text files with each line corresponding to a timestamp. Each line in the file contains the aforementioned measurements, which are used to compute the ground truth trajectory and 6-DoF motion of the vehicle. This information can be further used for calibration, data synchronization, and performance evaluation of various algorithms.
        """
        poses = []

        def latlon_to_mercator(lat, lon, s):
            """
            Converts latitude and longitude coordinates to Mercator coordinates (x, y) using the given scale factor.

            The Mercator projection is a widely used cylindrical map projection that represents the Earth's surface
            as a flat, rectangular grid, distorting the size of geographical features in higher latitudes.
            This function uses the scale factor 's' to control the amount of distortion in the projection.

            Args:
                lat (float): Latitude in degrees, range: -90 to 90.
                lon (float): Longitude in degrees, range: -180 to 180.
                s (float): Scale factor, typically the cosine of the reference latitude.

            Returns:
                list: A list containing the Mercator coordinates [x, y] in meters.
            """
            r = 6378137.0  # the Earth's equatorial radius in meters
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        # Compute the initial scale and pose based on the selected frames
        if selected_frames is None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = np.linalg.inv(pose_i)

        # Iterate through the OXTS data and compute the corresponding pose matrices
        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)  # (3,3)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)  # (4, 4)
            if pose_0_inv is None:
                pose_0_inv = np.linalg.inv(pose_i)

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    # If there is no tracking path specified, use the default path
    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, "oxts/data")
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    # If a tracking path is specified, use it to load OXTS data and compute the poses
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)  # (n_frames, 4, 4)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    # Return the poses, calibrations, and focal length
    return poses, calibrations, focal

def tracking_calib_from_txt(calibration_path):
    """
    Extract tracking calibration information from a KITTI tracking calibration file.

    This function reads a KITTI tracking calibration file and extracts the relevant
    calibration information, including projection matrices and transformation matrices
    for camera, LiDAR, and IMU coordinate systems.

    Args:
        calibration_path (str): Path to the KITTI tracking calibration file.

    Returns:
        dict: A dictionary containing the following calibration information:
            P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras.
            Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
            Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
            Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
    """
    # Read the calibration file
    f = open(calibration_path)
    calib_str = f.read().splitlines()

    # Process the calibration data
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    # Extract the projection matrices
    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    # Extract the transformation matrix for camera to rectified camera coordinates
    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect

    # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    return {
        "P0": P0,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Tr_cam2camrect": Tr_cam2camrect,
        "Tr_velo2cam": Tr_velo2cam,
        "Tr_imu2velo": Tr_imu2velo,
    }
    
def get_kittimot_calib(poses_velo_w_tracking, tracking_calibration, scene_no=None):
    exp = False
    camera_ls = [2,3]
    calib = {}
    
    # Get camera Poses   camare id: 02, 03
    for cam_i in camera_ls:
        transformation = np.eye(4)
        projection = tracking_calibration["P" + str(cam_i)]  # rectified camera coordinate system -> image
        K_inv = np.linalg.inv(projection[:3, :3])
        R_t = projection[:3, 3]
        t_crect2c = np.matmul(K_inv, R_t)
        transformation[:3, 3] = t_crect2c
        tracking_calibration["Tr_camrect2cam0" + str(cam_i)] = transformation
        
    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7)  ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9)  ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6)  ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        # roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    # cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration["Tr_cam2camrect"]
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = np.linalg.inv(Tr_cam2camrect)
    calib.update(Tr=tracking_calibration["Tr_velo2cam"])

    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration["Tr_camrect2cam0" + str(cam)]
        Tr_cam_i2camrect = np.linalg.inv(Tr_camrect2cam_i)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, Tr_cam_i2camrect)
        calib.update({'T_cam0_2_cam'+str(cam): np.linalg.inv(cam_i_cam0)})
        # calib.update({'T_02': np.linalg.inv(Tr_cam_i2camrect)@tracking_calibration["Tr_cam2camrect"]})
        calib.update({"P" + str(cam): tracking_calibration["P" + str(cam)]})
        

    return calib

def get_kitti360_calib(calib_path):
    intrinsic_loaded = False
    calib = {}
    # open file
    calib_cam2pose_file = os.path.join(calib_path, 'calib_cam_to_pose.txt')
    with open(calib_cam2pose_file) as f:
        cam2pose = f.read().splitlines()
    for line in cam2pose:
        line = line.split(' ')
        if line[0] == 'image_00:':
            cam_2_imu = np.eye(4) 
            cam_2_imu[:3,:4] = np.array([float(x) for x in line[1:-1]]).reshape(3,4)
    calib['cam0_2_imu'] = cam_2_imu
            
    calib_cam2velo_file = os.path.join(calib_path, 'calib_cam_to_velo.txt')
    with open(calib_cam2velo_file) as f:
        cam2velo = f.read().splitlines()
    cam0_2_velo = np.eye(4)
    line = cam2velo[0].split(' ')
    cam0_2_velo[:3,:4] = np.array([float(x) for x in line]).reshape(3,4)
    calib['Tr'] = np.linalg.inv(cam0_2_velo)
    calib['T_cam0_2_cam2'] = np.eye(4)
    
    intrinsic_file = os.path.join(calib_path, 'perspective.txt')
    with open(intrinsic_file) as f:
        intrinsics = f.read().splitlines()
    for line in intrinsics:
        line = line.split(' ')
        if line[0] == 'P_rect_00:':
            K = [float(x) for x in line[1:]]
            K = np.reshape(K, [3,4])
            calib['P2'] = K
            intrinsic_loaded = True
        elif line[0] == 'R_rect_00:':
            R_rect = np.eye(4) 
            R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            calib['R_rect'] = R_rect
        elif line[0] == "S_rect_00:":
            calib['width'] = int(float(line[1]))
            calib['height'] = int(float(line[2]))
    assert(intrinsic_loaded==True)
    return calib
    
def get_poses_kitti360(calib, pose_file):
    # load poses
    poses = np.loadtxt(pose_file)
    frames = poses[:,0].astype(int)
    poses = np.reshape(poses[:,1:],[-1,3,4])
    cam2world = {}
    for frame, pose in zip(frames, poses): 
        pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
        # consider the rectification for perspective cameras
        cam2world[frame] = np.matmul(np.matmul(pose, calib['cam0_2_imu']),np.linalg.inv(calib['R_rect']))
    return cam2world
    
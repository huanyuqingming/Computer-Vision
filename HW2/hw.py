import cv2
import numpy as np
import glob
from tqdm import tqdm

np.set_printoptions(suppress=True, precision=8)

def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob(f"{image_directory}/*.jpg")]

def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        width, height = pattern_size
        world_point = []
        for i in range(height):
            for j in range(width):
                world_point.append([j, i, 0])

        return np.array(world_point, dtype=np.float32)
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        ret, corners = cv2.findChessboardCorners(image, pattern_size)
        if ret:
            corners = corners.reshape(-1, 2)
        return corners

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in tqdm(images, desc="Detecting corners"):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            # Add image corners
            corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4))
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))

    return np.array(world_points), np.array(image_points)

def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_images = len(world_points)
    HA = []
    B = []
    K = np.zeros((4, 4))
    P = []

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    for i in range(num_images):
        A = []
        world_point, image_point = world_points[i], image_points[i]
        num_points = len(world_point)

        for j in range(num_points):
            x, y = world_point[j]
            u, v = image_point[j]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

        A = np.array(A)
        eig_value, eig_vector = np.linalg.eig(A.T @ A)
        h = eig_vector[:, np.argmin(eig_value)]
        H = h.reshape(3, 3)
        P.append(H)

        h11, h12, h13, h21, h22, h23, h31, h32, h33 = h
        HA.append([h11 * h11 - h12 * h12, 2 * (h11 * h21 - h12 * h22), 2 * (h11 * h31 - h12 * h32), h21 * h21 - h22 * h22, 2 * (h21 * h31 - h22 * h32), h31 * h31 - h32 * h32])
        HA.append([h11 * h12, h11 * h22 + h12 * h21, h11 * h32 + h12 * h31, h21 * h22, h21 * h32 + h22 * h31, h31 * h32])

    HA = np.array(HA)
    eig_value, eig_vector = np.linalg.eig(HA.T @ HA)
    b = eig_vector[:, np.argmin(eig_value)]
    B = np.array([[b[0], b[1], b[2]], [b[1], b[3], b[4]], [b[2], b[4], b[5]]])

    K = np.linalg.inv(np.linalg.cholesky(B).T)
    K /= K[2, 2]
    
    assert (K[0, 0] > 0 and K[1, 1] > 0 and K[2, 2] > 0), "The diagonal elements of K must be positive"
    # Please ensure that the diagonal elements of K are positive
    
    return K, P

# Main process
image_path = 'Sample_Calibration_Images'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (31, 23)  # The patter`n size of the chessboard 

world_points, image_points = find_image_points(images, pattern_size)

# camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)
camera_matrix, camera_extrinsics = calibrate_camera(world_points[:, :, :-1], image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)

def test(images, world_points, image_points):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    # TODO
    image_size = images[0].shape[1], images[0].shape[0]
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, image_size, None, None)
        
    print("Camera Calibration Matrix by OpenCV:")
    print(camera_matrix)

def reprojection_error(world_points, image_points, camera_matrix):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image
    errors = []
    world_points = np.array(world_points)[:, :, :-1]
    num_images = len(world_points)

    for i in range(num_images):
        expand_vector = np.ones((world_points[i].shape[0], 1), dtype=np.float32)
        object_points = np.append(world_points[i], expand_vector, axis=1)

        projected_points = camera_matrix[i] @ object_points.T
        projected_points /= projected_points[2]

        norm_error = np.linalg.norm(image_points[i].T - projected_points[:2], axis=0)
        errors.append(np.mean(norm_error))
        
    print("Reprojection Error:")
    print(errors)

test(images, world_points, image_points)
reprojection_error(world_points, image_points, camera_extrinsics)

#!/usr/bin/env python3
import sys
import cv2
import numpy as np


def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    # raise NotImplementedError  #TODO
    def convolve(image, kernel):
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape
        convolved_image = np.zeros((image_height, image_width), dtype=float)
        for i in range(image_height - kernel_height + 1):
            for j in range(image_width - kernel_width + 1):
                convolved_image[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
        return convolved_image
  
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    grad_x = convolve(image, x_kernel)
    grad_y = convolve(image, y_kernel)
    edge_image = np.sqrt(np.square(grad_x) + np.square(grad_y))
    edge_image = 255 * edge_image / np.max(edge_image)
    
    return edge_image


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    # raise NotImplementedError  #TODO
    thresh_edge_image = np.where(edge_image >= edge_thresh, True, False)
    H, W = edge_image.shape
    R = len(radius_values)
    accum_array = np.zeros((R, H, W), dtype=int)

    y_indices, x_indices = np.nonzero(thresh_edge_image)
    y_indices = y_indices.reshape(-1, 1)
    x_indices = x_indices.reshape(-1, 1)

    for r_idx, r in enumerate(radius_values):
        thetas = np.arange(360)
        a_offsets = (r * np.cos(np.radians(thetas))).astype(int)
        b_offsets = (r * np.sin(np.radians(thetas))).astype(int)

        a = x_indices - a_offsets
        b = y_indices - b_offsets

        valid_mask = (0 <= a) & (a < W) & (0 <= b) & (b < H)

        for i in range(len(y_indices)):
            accum_array[r_idx, b[i][valid_mask[i]], a[i][valid_mask[i]]] += 1

    return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    # raise NotImplementedError  #TODO
    circles_attribute = np.where(accum_array >= hough_thresh, True, False)
    circles = []
    for r_idx, r in enumerate(radius_values):
        y_indices, x_indices = np.nonzero(circles_attribute[r_idx])
        for i in range(len(y_indices)):
            is_overlap = False
            EPS = 5
            for circle in circles:
                if (x_indices[i] - circle[2]) ** 2 + (y_indices[i] - circle[1]) ** 2 <= EPS ** 2:
                    is_overlap = True
                    break
            if not is_overlap:
                circles.append((int(r), int(y_indices[i]), int(x_indices[i])))

    circles_image = image.copy()
    for circle in circles:
        cv2.circle(circles_image, (circle[2], circle[1]), circle[0], (0, 255, 0), 2)

    return circles, circles_image


def main(argv):
    img_name = argv[0]
    edge_thresh = int(argv[1])
    radius_values = range(int(argv[2]), int(argv[3]) + 1)
    hough_thresh = int(argv[4])

    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = detect_edges(gray_img)
    cv2.imwrite('output/' + img_name + "_edges_sobel.png", edge_img)
    thresh_edge_img, accum_array = hough_circles(edge_img, edge_thresh, radius_values)
    cv2.imwrite('output/' + img_name + "_edges.png", 255 * thresh_edge_img)
    circles, circles_image = find_circles(img, accum_array, radius_values, hough_thresh)
    cv2.imwrite('output/' + img_name + "_circles.png", circles_image)
    print(circles)

if __name__ == '__main__':
    #TODO
    main(sys.argv[1:])
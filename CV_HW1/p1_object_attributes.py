#!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.where(gray_image >= thresh_val, 255, 0)
  return binary_image

def label(binary_image):
  # TODO
  height, width = binary_image.shape
  labeled_image = np.zeros((height, width), dtype=np.uint8)
  current_label = 1
  
  def dfs(x, y):
    stack = [(x, y)]
    while stack:
      cx, cy = stack.pop()
      if cx < 0 or cx >= height or cy < 0 or cy >= width:
        continue
      if binary_image[cx, cy] == 255 and labeled_image[cx, cy] == 0:
        labeled_image[cx, cy] = current_label
        stack.append((cx + 1, cy))
        stack.append((cx - 1, cy))
        stack.append((cx, cy + 1))
        stack.append((cx, cy - 1))

  for i in range(height):
      for j in range(width):
          if binary_image[i, j] == 255 and labeled_image[i, j] == 0:
              dfs(i, j)
              current_label += 1

  for label in range(1, current_label):
      labeled_image[labeled_image == label] = int(255 * label / current_label)
  
  return labeled_image

def get_attribute(labeled_image):
  # TODO
  height, width = labeled_image.shape
  attribute_list = []

  def get_position(area):
    x = (1.0 / len(area)) * sum([p[0] for p in area])
    y = (1.0 / len(area)) * sum([p[1] for p in area])
    return x, y

  def get_orientation(area, x, y):
    a = sum([(p[0] - x) ** 2 for p in area])
    b = 2 * sum([(p[0] - x) * (p[1] - y) for p in area])
    c = sum([(p[1] - y) ** 2 for p in area])
    theta = 0.5 * np.arctan2(b, (a - c))
    return theta, a, b, c

  def get_roundedness(theta, a, b, c):
    Emin = a * (np.sin(theta) ** 2) - b * np.sin(theta) * np.cos(theta) + c * (np.cos(theta) ** 2)
    Emax = a * (np.cos(theta) ** 2) + b * np.sin(theta) * np.cos(theta) + c * (np.sin(theta) ** 2)
    return Emin / Emax

  for label in range(1, 256):
    area = np.argwhere(labeled_image == label)
    for pixel in area:
      pixel[0], pixel[1] = pixel[1], pixel[0]
    if len(area) == 0:
      continue
    x, y = get_position(area)
    theta, a, b, c = get_orientation(area, x, y)
    roundedness = get_roundedness(theta, a, b, c)
    attribute = {
      'position': {'x': float(x), 'y': float(height-1-y)},
      'orientation': float(theta),
      'roundedness': float(roundedness)
    }
    attribute_list.append(attribute)

  # # 输出标记有每个连通域中心的图像
  # signed_image = labeled_image.copy()
  # for attribute in attribute_list:
  #   x, y = attribute['position']['x'], height-1-attribute['position']['y']
  #   x, y = int(x), int(y)
  #   cv2.circle(signed_image, (x, y), 5, 255, -1)
  #   # 在图像中添加方向箭头
  #   theta = attribute['orientation']
  #   x1 = x + 50 * np.cos(theta)
  #   y1 = y + 50 * np.sin(theta)
  #   x1, y1 = int(x1), int(y1)
  #   cv2.arrowedLine(signed_image, (x, y), (x1, y1), 255, 1)
  #   x1 = x + 50 * np.sin(theta)
  #   y1 = y + 50 * -np.cos(theta)
  #   x1, y1 = int(x1), int(y1)
  #   cv2.arrowedLine(signed_image, (x, y), (x1, y1), 255, 1)
  # cv2.imwrite('output/signed_image.png', signed_image)

  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  print(attribute_list)


if __name__ == '__main__':
  main(sys.argv[1:])

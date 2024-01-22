import cv2
import numpy as np


class CalibratorPlane:
    def __init__(self):
        self.size = (0, 0)
        self.quad_image = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.origin = (0, 0)
        self.offset = (0, 0)

    def initialize(self, size, quad_image):
        self.size = size
        self.quad_image = quad_image
        self.origin = self.calculate_origin()

    def calculate_origin(self):
        # Assuming the origin is the bottom-left point of the quad
        return self.quad_image[3]

    def set_origin(self, p):
        # Set the origin of the calibration plane
        self.origin = p

    def update(self, quad_image):
        self.quad_image = quad_image

    def transform(self, point, origin):
        # Transform a point based on the calibration plane
        transformed_point = np.array(point) - np.array(origin) + np.array(self.offset)
        return tuple(transformed_point)

    def get_homography_matrix(self):
        # Calculate the homography matrix based on the user input quad
        src = np.array(self.quad_image, dtype=np.float32)
        dst = np.array([(0, 0), (self.size[0], 0), self.size, (0, self.size[1])], dtype=np.float32)

        return cv2.getPerspectiveTransform(src, dst)

    def transform_to_quad(self, point):
        # Transform a point to the user input quad
        homography_matrix = self.get_homography_matrix()
        transformed_point = cv2.perspectiveTransform(np.array([point], dtype=np.float32), np.linalg.inv(homography_matrix))
        return tuple(transformed_point[0][0])

class CalibratorLine:
    def __init__(self, quad):
        self.quad = quad  # Quadrilateral representing the calibration line
        self.origin = self.quad[0]  # Initial origin is the first point of the quad

    def set_origin(self, p):
        # Set the origin of the calibration line
        self.origin = p

    def transform(self, point):
        # Transform a point based on the calibration line
        # This is a simple example where the X-coordinate is transformed
        transformed_x = point[0] - self.origin[0]
        transformed_y = point[1]  # Y-coordinate remains unchanged
        return (transformed_x, transformed_y)
    
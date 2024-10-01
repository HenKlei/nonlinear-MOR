import numpy as np
import cv2


class Line:
    def __init__(self, starting_point, end_point):
        self.starting_point = starting_point
        self.end_point = end_point
        self.direction = self.end_point - self.starting_point

    def length(self):
        return np.linalg.norm(self.starting_point - self.end_point)

    def get_uniform_landmarks(self, n):
        if n <= 1:
            n = 2
        return [self.starting_point + i / (n - 1) * self.direction for i in range(n)]

    def nearest_point_on_line(self, point):
        r0 = self.starting_point
        r1 = self.end_point
        d = np.linalg.norm(self.direction)
        r01u = self.direction / d
        r = point - r0
        rid = np.dot(r, r01u)
        ri = r01u * rid
        lpt = r0 + ri
        if rid > d:
            return r1
        if rid < 0:
            return r0
        return lpt

    def point_close_to_line(self, point, threshold=1e-1):
        proj = self.nearest_point_on_line(point)
        if np.linalg.norm(point - proj) > threshold:
            return False
        return True

    def __str__(self):
        return (f"Line: ({self.starting_point[0]}, {self.starting_point[1]}) "
                f"to ({self.end_point[0]}, {self.end_point[1]})")


def place_landmarks_on_edges(input_img, num_landmarks,
                             edge_detection_parameters={"low_threshold": 50, "high_threshold": 150},
                             perform_Gaussian_blurring=False, blur_kernel_size=5,
                             hough_parameters={
                                 "rho": 1,  # distance resolution in pixels of the Hough grid
                                 "theta": np.pi / 180,  # angular resolution in radians of the Hough grid
                                 "threshold": 30,  # minimum number of votes (intersections in Hough grid cell)
                                 "minLineLength": 10,  # minimum number of pixels making up a line
                                 "maxLineGap": 20},  # maximum gap in pixels between connectable line segments
                             remove_parallel_lines=True, remove_parallel_lines_threshold=1,
                             mins=np.array([0., 0.]), maxs=np.array([1., 1.])
                             ):
    img = input_img.copy()
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val) * 255.
    img = np.asarray(img, dtype=np.uint8)

    if perform_Gaussian_blurring:
        img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

    edges = cv2.Canny(img, edge_detection_parameters["low_threshold"], edge_detection_parameters["high_threshold"])

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, **hough_parameters)

    img = np.asarray(img, dtype=float)
    # print(f"Number of lines: {len(lines)}")

    line_objects = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_objects.append(Line(np.array([x1, y1]), np.array([x2, y2])))

    if remove_parallel_lines:
        new_line_objects = []
        for l1 in line_objects:
            for l2 in line_objects:
                if not l1 == l2:
                    if l1.point_close_to_line(l2.starting_point, threshold=remove_parallel_lines_threshold):
                        if l2.point_close_to_line(l1.starting_point, threshold=remove_parallel_lines_threshold):
                            new_line_object = Line(l1.starting_point.copy(), l2.starting_point.copy())
                            new_line_objects.append(new_line_object)
                            l2.starting_point = l1.starting_point.copy()
                            l1.starting_point = new_line_object.end_point.copy()
                        elif l2.point_close_to_line(l1.end_point, threshold=remove_parallel_lines_threshold):
                            new_line_object = Line(l2.starting_point.copy(), l1.end_point.copy())
                            new_line_objects.append(new_line_object)
                            l2.starting_point = l1.end_point.copy()
                            l1.end_point = new_line_object.starting_point.copy()
                    if l1.point_close_to_line(l2.end_point, threshold=remove_parallel_lines_threshold):
                        if l2.point_close_to_line(l1.starting_point, threshold=remove_parallel_lines_threshold):
                            new_line_object = Line(l1.starting_point.copy(), l2.end_point.copy())
                            new_line_objects.append(new_line_object)
                            l2.end_point = l1.starting_point.copy()
                            l1.starting_point = new_line_object.end_point.copy()
                        elif l2.point_close_to_line(l1.end_point, threshold=remove_parallel_lines_threshold):
                            new_line_object = Line(l2.end_point.copy(), l1.end_point.copy())
                            new_line_objects.append(new_line_object)
                            l2.end_point = l1.end_point.copy()
                            l1.end_point = new_line_object.starting_point.copy()
        line_objects.extend(new_line_objects)

    total_line_length = 0.
    for line_object in line_objects:
        total_line_length += line_object.length()
    # print(f"Total line length: {total_line_length}")

    landmarks = []
    for line_object in line_objects:
        landmarks.extend(line_object.get_uniform_landmarks(int(num_landmarks*line_object.length()/total_line_length)))
    true_landmarks = []
    for l in landmarks:
        landmark_accepted = True
        for i in range(img.ndim):
            l[i] /= img.shape[i]
            if not (mins[i] <= l[i] <= maxs[i]):
                landmark_accepted = False
        if landmark_accepted:
            true_landmarks.append(l)
    landmarks = np.array(true_landmarks)
    #import matplotlib.pyplot as plt
    #plt.imshow(img, origin="lower")
    #plt.scatter(landmarks[..., 0], landmarks[..., 1], c="red")
    #plt.show()
    #for i in range(img.ndim):
    #    landmarks[..., i] /= img.shape[i]
    return landmarks

import cv2


class Box:

    DRAW_COLOR = (77, 255, 9)
    DRAW_THICKNESS = 1

    def __init__(self, box_data, score):
        if box_data is None:
            self.left = None
            self.right = None
            self.top = None
            self.bottom = None
            self.cx = None
            self.cy = None
            self.area = None
            self.score = None
        else:
            self.score = score
            self.left = box_data[1]
            self.right = box_data[3]
            self.top = box_data[0]
            self.bottom = box_data[2]
            self.cx = self.left + (self.right - self.left) / 2
            self.cy = self.top + (self.bottom - self.top) / 2
            self.area = self.__calc_area()

    def __calc_area(self):
        return (self.right - self.left) * (self.bottom - self.top)

    def manhattan_to_box(self, other, frame_shape=None):
        return self.manhattan_to_point(point=(other.cx, other.cy), frame_shape=frame_shape)

    def manhattan_to_point(self, point, frame_shape=None):
        x, y = point
        if frame_shape is None:
            dy = y - self.cy
            dx = x - self.cx
        else:
            dy = (y - self.cy) * frame_shape[0]
            dx = (x - self.cx) * frame_shape[1]
        return dx * dx + dy * dy

    def manhattan_to_contour(self, contour, frame_shape):
        x, y, w, h = cv2.boundingRect(contour)
        # Normalize the distances of the contour
        cy = (y + h // 2) / frame_shape[0] + self.top
        cx = (x + w // 2) / frame_shape[1] + self.left
        return self.manhattan_to_point(point=(cx, cy), frame_shape=None)

    def expand(self, percentage):
        percentage -= 1.0  # Take out 100% to calculate the rest
        dw = int((self.right - self.left) * percentage) // 2
        dh = int((self.bottom - self.top) * percentage) // 2
        self.left   = max(self.left - dw, 0)
        self.right  = min(self.right + dw, 1.0)
        self.top    = max(self.top - dh, 0)
        self.bottom = min(self.bottom + dh, 1.0)
        self.area = self.__calc_area()

    def get_roi(self, frame):  # Careful, returns a view
        h = frame.shape[0]
        w = frame.shape[1]
        return frame[int(self.top * h):int(self.bottom * h),
                     int(self.left * w):int(self.right * w)]

    def draw(self, frame):
        h = frame.shape[0]
        w = frame.shape[1]
        if self.left and self.right and self.top and self.bottom:
            p1 = (int(self.left * w), int(self.top * h))
            p2 = (int(self.right * w), int(self.bottom * h))
            cv2.rectangle(frame, p1, p2, Box.DRAW_COLOR, Box.DRAW_THICKNESS)

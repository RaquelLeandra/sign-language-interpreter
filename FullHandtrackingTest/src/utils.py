

def manhattan(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return dx * dx + dy * dy


def manhattan_boxes(box1, box2, shape):
    h = shape[0]
    w = shape[1]
    dx = (box2.cx - box1.cx) * w
    dy = (box2.cy - box1.cy) * h
    return dx * dx + dy * dy


def normalized_manhattan_boxes(box1, box2):
    dx = box2.cx - box1.cx
    dy = box2.cy - box1.cy
    return dx * dx + dy * dy

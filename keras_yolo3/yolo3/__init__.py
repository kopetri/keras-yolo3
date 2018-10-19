import numpy


def get_yolo3_anchors():
    anchors = numpy.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
    return numpy.reshape(anchors, (9, 2))

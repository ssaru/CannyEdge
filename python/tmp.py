>>> import numpy as np
>>> np.set_printoptions(precision=2)

Virtual Image
>>> source = np.array([[1, 2 ,3 ,4 ,5], \
                       [6, 7, 8, 9, 10], \
                       [1, 2 ,3 ,4 ,5], \
                       [6, 7, 8, 9, 10], \
                       [1, 2 ,3 ,4 ,5]], dtype=np.uint8)

Average Filter
>>> filter = np.array([[1/9, 1/9, 1/9], \
                       [1/9, 1/9, 1/9], \
                       [1/9, 1/9, 1/9]], dtype=np.float64)
>>> stride = 1

>>> correlate2d(source=source, filter=filter, stride=stride)
array([[3.67, 4.67, 5.67],
       [5.33, 6.33, 7.33],
       [3.67, 4.67, 5.67]])

# Refer OpenCV Document
# https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
gaussian_filter = np.array([[2,   4,  5,  4,  2],
                            [4,   9,  12, 9,  4],
                            [5,   12, 15, 12, 5],
                            [4,   9,  12, 9,  4],
                            [2,   4,  5,  4,  2]]) / 159
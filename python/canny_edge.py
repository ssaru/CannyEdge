import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union

from PIL import Image


def rgb2gray(source: np.ndarray):
    """
    RGB image convert to grayscale image

    Args:
        source (np.ndarray): RGB numpy array image as following [width, height, channels].
                             channels ordered by [R, G, B]

    Returns:
        (np.ndarray): grayscale image as following [width, height]

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    >>> source = np.array([[[1, 1, 1], [1, 1, 1,], [1, 1, 1]]])
    >>> rgb2gray(source)
    array([[1., 1., 1.]])
    """
    if len(source.shape) == 2:
        return source

    return source[:, :, 0] * 0.2989 + source[:, :, 1] * 0.5870 + source[:, :, 2] * 0.1140


def pad2d(source: np.ndarray,
          pad_size: Tuple[int, int],
          pad_value: int = 0) -> np.ndarray:
    """
    Args:
        source      (np.ndarray) : 2-D array for applying padding. data type should be uint8
        pad_size    (Tuple[int, int]) : pad size as following (width, height)
        pad_value   (int) : pad value

    Returns:
        (np.ndarray) : result of padding to source array

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    Set input parameter
    >>> source = np.array([[1 ,1 ,1], \
                           [1, 1, 1], \
                           [1, 1, 1]], dtype=np.uint8)

    >>> pad_size = (2, 2)
    >>> pad_value = 0

    Expected value when called function
    >>> pad2d(source=source, pad_size=pad_size, pad_value=pad_value)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """

    assert len(source.shape) == 2, f"Source Only supported 2-D array. : {source.shape}"
    assert len(pad_size) == 2, f"Pad Size Only supported 2-D array. : {pad_size}"

    pad_width, pad_height = pad_size

    source_list: List = source.tolist()

    for i in range(pad_width):
        for row in source_list:
            row.append(pad_value)
            row.insert(0, pad_value)

    sequence_length = len(source_list[0])
    pad_sequence = [pad_value for i in range(sequence_length)]

    for i in range(pad_height):
        source_list.append(pad_sequence)
        source_list.insert(0, pad_sequence)

    return np.asarray(source_list, np.uint8)


def hadamard_product2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    hadamard product 2d operation
    refer here : https://en.wikipedia.org/wiki/Hadamard_product_(matrices)

    Args:
        x1 (np.ndarray): 2-D array as following [width, height]
        x2 (np.ndarray): 2-D array as following [width, height]

    Return:
        (np.ndarray)

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    Set input parameter
    >>> x1 = np.array([[1, 2, 3], \
                       [4, 5, 6], \
                       [7, 8, 9]])

    >>> x2 = np.array([[1, 2, 3], \
                       [4, 5, 6], \
                       [7, 8, 9]])

    Expected value when called function
    >>> hadamard_product2d(x1, x2)
    array([[ 1.,  4.,  9.],
           [16., 25., 36.],
           [49., 64., 81.]])
    """

    assert x1.shape == x2.shape, \
        f"hadamard product require shape of two argument should be same. x1 shape : {x1.shape}, x2 shape : {x2.shape}"
    assert len(x1.shape) == 2, f"It only support 2-D hadamard product. x1.shape : {x1.shape}"

    output = np.zeros(x1.shape)

    rows, columns = x1.shape
    for row in range(rows):
        for column in range(columns):
            output[row][column] = x1[row][column] * x2[row][column]

    return output


def correlate2d(source: np.array,
                mask: np.ndarray,
                stride: int) -> np.ndarray:
    """
    Correlate operator for Image Processing.
    refer here : https://en.wikipedia.org/wiki/Cross-correlation

    Args:
        source (np.ndarray): 2-D array as following [width, height]
        mask   (np.ndarray): 2-D array as following [width, height]
        stride (int): stride value

    Returns:
        (np.ndarray):

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    Set input parameter
    Virtual image -> source
    >>> source = np.array([[1, 2 ,3 ,4 ,5], \
                           [6, 7, 8, 9, 10], \
                           [1, 2 ,3 ,4 ,5], \
                           [6, 7, 8, 9, 10], \
                           [1, 2 ,3 ,4 ,5]], dtype=np.uint8)

    Average filter -> filter
    >>> mask = np.array([[1/9, 1/9, 1/9], \
                         [1/9, 1/9, 1/9], \
                         [1/9, 1/9, 1/9]], dtype=np.float64)
    >>> stride = 1

    Expected value when called function
    >>> correlate2d(source=source, mask=mask, stride=stride)
    array([[3.67, 4.67, 5.67],
           [5.33, 6.33, 7.33],
           [3.67, 4.67, 5.67]])
    >>> stride = 2
    >>> correlate2d(source=source, mask=mask, stride=stride)
    array([[3.67, 5.67],
           [3.67, 5.67]])
    """

    assert len(source.shape) == 2, "Only supported 2-D array"
    assert len(mask.shape) == 2, "Only supported 2-D array"
    assert stride > 0, f"Stride value should be positive : {stride}"

    source_width, source_height = source.shape
    mask_width, mask_height = mask.shape

    # Output size = {(((Source width) - (Mask width))  / Stride) + 1,
    #                   (((Source height) - (Mask height))  / Stride) + 1}
    output_width = int(((source_width - mask_width) / stride) + 1)
    output_height = int(((source_height - mask_height) / stride) + 1)

    assert ((source_width - mask_width) % stride) == 0, f"Output Width should be int : {output_width}"
    assert ((source_height - mask_height) % stride) == 0, f"Output Height should be int : {output_height}"

    output = np.zeros((output_width, output_height))

    source_y_idx = 0
    for output_y_idx in range(output_height):
        source_x_idx = 0
        y_idx = source_y_idx * stride

        for output_x_idx in range(output_width):
            x_idx = source_x_idx * stride
            sliced_source = source[y_idx: y_idx + mask_height, x_idx: x_idx + mask_width]
            output[output_y_idx][output_x_idx] = np.sum(hadamard_product2d(sliced_source, mask)).squeeze()
            source_x_idx += 1

        source_y_idx += 1

    return output


def reflection_y(source: np.ndarray):
    """
    Reflection Transformation at y-axis

    Args:
        source (np.ndarray) : 2-D array as following [width, height]

    Returns:
        (np.ndarray)

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    Set input parameter
    >>> source = np.array([[1, 2 ,3 ,4 ,5], \
                           [6, 7, 8, 9, 10], \
                           [1, 2 ,3 ,4 ,5], \
                           [6, 7, 8, 9, 10], \
                           [1, 2 ,3 ,4 ,5]], dtype=np.uint8)

    Expected value when called function
    >>> reflection_y(source=source)
    array([[ 5,  4,  3,  2,  1],
           [10,  9,  8,  7,  6],
           [ 5,  4,  3,  2,  1],
           [10,  9,  8,  7,  6],
           [ 5,  4,  3,  2,  1]], dtype=uint8)

    >>> filter = np.array([[1, 2, 3], \
                           [4, 5, 6], \
                           [7, 8, 9]])

    >>> reflection_y(source=filter)
    array([[3, 2, 1],
           [6, 5, 4],
           [9, 8, 7]])

    """

    assert len(source.shape) == 2, f"It's Only supported 2-D array : {source.shape}"

    rows, columns = source.shape
    output = copy.deepcopy(source)

    for y_idx in range(rows):
        for x_idx in range(columns//2):
            output[y_idx][x_idx] = source[y_idx][columns - x_idx - 1]
            output[y_idx][columns - x_idx - 1] = source[y_idx][x_idx]

    return output


def reflection_x(source: np.ndarray):
    """
    Reflection Transformation at x-axis

    Args:
        source (np.ndarray) : 2-D array as following [width, height]

    Returns:
        (np.ndarray)

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    Set input parameter
    >>> source = np.array([[1, 2, 3, 4, 5], \
                           [1, 2, 3, 4, 5], \
                           [0, 0, 0, 0, 0], \
                           [6, 7, 8, 9, 10], \
                           [6, 7, 8, 9, 10]], dtype=np.uint8)

    Expected value when called function
    >>> reflection_x(source=source)
    array([[ 6,  7,  8,  9, 10],
           [ 6,  7,  8,  9, 10],
           [ 0,  0,  0,  0,  0],
           [ 1,  2,  3,  4,  5],
           [ 1,  2,  3,  4,  5]], dtype=uint8)

    """

    assert len(source.shape) == 2, f"It's Only supported 2-D array : {source.shape}"

    rows, columns = source.shape
    output = copy.deepcopy(source)

    for y_idx in range(rows//2):
        output[y_idx] = source[rows - y_idx - 1]
        output[rows - y_idx - 1] = source[y_idx]

    return output


def convolve2d(source: np.ndarray,
               mask: np.ndarray,
               stride: int = 1,
               pad_size: Tuple[int, int] = (0, 0),
               pad_value: int = 0):
    """

    Args:
        source (np.ndarray): 2-D array as following [width, height]
        mask   (np.ndarray): 2-D array as following [width, height]
        stride (int): stride value
        pad_size (Tuple[int, int]): padding size
        pad_value (int): fill specific value in padding

    Returns:
        (np.ndarray)

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    Set input parameter
    >>> source = np.array([[1, 2, 3], \
                           [4, 5, 6], \
                           [7, 8, 9]], dtype=np.uint8)

    >>> mask = np.array([[1, 2, 3], \
                         [4, 5, 6], \
                         [7, 8, 9]])
    >>> stride = 1
    >>> pad_size = (0, 0)
    >>> pad_value = 0

    >>> convolve2d(source=source, mask=mask, stride=stride, pad_size=pad_size, pad_value=pad_value)
    array([[165.]])

    """

    source = pad2d(source=source,
                   pad_size=pad_size,
                   pad_value=pad_value)

    mask = reflection_y(mask)
    mask = reflection_x(mask)

    return correlate2d(source=source, mask=mask, stride=1)


def filter(source: np.ndarray,
           mask: np.ndarray,
           stride: int = 1,
           pad_size=(0, 0),
           pad_value : int = 0):
    """
    Filtering for image processing

    Args:
        source      (numpy.ndarray): 2-D array as following [width, height]
        mask        (numpy.ndarray): 2-D array as following [width, height]
        stride      (int): stride value
        pad_size    (Tuple[int, int]): padding size
        pad_value   (int): fill specific value in padding

    Returns:
        (np.ndarray) : numpy image array as following [width, height]

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    >>> source = np.array([[1, 1, 1, 1, 1], \
                           [1, 1, 1, 1, 1], \
                           [1, 1, 1, 1, 1], \
                           [1, 1, 1, 1, 1], \
                           [1, 1, 1, 1, 1]], dtype=np.uint8)
    >>> mask = np.array([[2,   4,  5,  4,  2], \
                         [4,   9,  12, 9,  4], \
                         [5,   12, 15, 12, 5], \
                         [4,   9,  12, 9,  4], \
                         [2,   4,  5,  4,  2]]) / 159
    >>> pad_size = (0, 0)
    >>> pad_value = 0
    >>> stride = 1

    >>> filter(source=source, mask=mask, stride=stride, pad_size=pad_size, pad_value=pad_value)
    array([[1.]])
    """

    return convolve2d(source=source, mask=mask, stride=stride, pad_size=pad_size, pad_value=pad_value)


def gaussian_smoothing(source: np.ndarray,
                       stride: int = 1,
                       pad_size=(0, 0),
                       pad_value : int = 0):
    """
    Gaussian Smoothing
    Refer here: https://en.wikipedia.org/wiki/Gaussian_blur

    Args:
        source      (numpy.ndarray): 2-D array as following [width, height]
        stride      (int): stride value
        pad_size    (Tuple[int, int]): padding size
        pad_value   (int): fill specific value in padding

    Returns:
        (np.ndarray) : numpy image array as following [width, height]
    """

    gaussian_filter = np.array([[2, 4, 5, 4, 2],
                                [4, 9, 12, 9, 4],
                                [5, 12, 15, 12, 5],
                                [4, 9, 12, 9, 4],
                                [2, 4, 5, 4, 2]]) / 159

    return filter(source=source, mask=gaussian_filter, stride=stride, pad_size=pad_size, pad_value=pad_value)


def sobel_x(source: np.ndarray,
            stride: int = 1,
            pad_size=(0, 0),
            pad_value : int = 0):
    """
    Sobel Operator
    Refer here: https://en.wikipedia.org/wiki/Sobel_operator

    Args:
        source      (numpy.ndarray): 2-D array as following [width, height]
        stride      (int): stride value
        pad_size    (Tuple[int, int]): padding size
        pad_value   (int): fill specific value in padding

    Returns:
        (np.ndarray) : numpy image array as following [width, height]
    """

    sobelmask_gx = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

    return filter(source=source, mask=sobelmask_gx, stride=stride, pad_size=pad_size, pad_value=pad_value)


def sobel_y(source: np.ndarray,
            stride: int = 1,
            pad_size=(0, 0),
            pad_value : int = 0):
    """
    Sobel Operator
    Refer here: https://en.wikipedia.org/wiki/Sobel_operator

    Args:
        source      (numpy.ndarray): 2-D array as following [width, height]
        stride      (int): stride value
        pad_size    (Tuple[int, int]): padding size
        pad_value   (int): fill specific value in padding

    Returns:
        (np.ndarray) : numpy image array as following [width, height]
    """

    sobelmask_gy = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]])

    return filter(source=source, mask=sobelmask_gy, stride=stride, pad_size=pad_size, pad_value=pad_value)


def gradient_vector_field(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Gradient Vector Field
    Refer here: https://en.wikipedia.org/wiki/Vector_field
                https://angeloyeo.github.io/2019/08/25/gradient.html

    Args:
        gx (np.ndarray): x direction gradient as 2-D array
        gy (np.ndarray): y direction gradient as 2-D array

    Returns:
         (np.ndarray): gradient vector field as 3-D array

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    >>> gx = np.array([[1, 2, 3], \
                       [4, 5, 6], \
                       [7, 8, 9]])
    >>> gy = np.array([[1, 2, 3], \
                       [4, 5, 6], \
                       [7, 8, 9]])
    >>> gradient_vector_field(gx, gy)
    array([[[1., 1.],
            [2., 2.],
            [3., 3.]],
    <BLANKLINE>
           [[4., 4.],
            [5., 5.],
            [6., 6.]],
    <BLANKLINE>
           [[7., 7.],
            [8., 8.],
            [9., 9.]]])
    """
    assert gx.shape == gy.shape, f"shape of two argument should be same : gx->{gx.shape}, gy->{gy.shape}"

    output_shape = list(gx.shape)
    output_shape.append(2)
    output_shape = tuple(output_shape)
    output_width, output_height, output_channels = output_shape

    output = np.zeros(output_shape)

    for x_idx in range(output_width):
        for y_idx in range(output_height):
            output[x_idx][y_idx][0] = gx[x_idx][y_idx]
            output[x_idx][y_idx][1] = gy[x_idx][y_idx]

    return output


def norm_vector_field(vector_field: np.ndarray):
    """
    norm of vector field

    Args:
        vector_field (np.ndarray): 3-D array vector field as following [w, h, c]

    Returns:
        (np.ndarray): 2-D array norm as following [w, h]

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    >>> vector_field = np.array([[[1., 1.], [2., 2.], [3., 3.]], \
                                 [[4., 4.], [5., 5.], [6., 6.]], \
                                 [[7., 7.], [8., 8.], [9., 9.]]])

    >>> norm_vector_field(vector_field=vector_field)
    array([[ 1.41,  2.83,  4.24],
           [ 5.66,  7.07,  8.49],
           [ 9.9 , 11.31, 12.73]])
    """

    output_width, output_height, _ = vector_field.shape
    output = np.zeros((output_width, output_height))

    for x_idx in range(output_width):
        for y_idx in range(output_height):
            if (vector_field[x_idx][y_idx][0] == 0) and (vector_field[x_idx][y_idx][1] == 0):
                output[x_idx][y_idx] = 0
            else:
                output[x_idx][y_idx] = np.sqrt((vector_field[x_idx][y_idx][0] ** 2) + (vector_field[x_idx][y_idx][1] ** 2))

    return output


def direction_vector_field(vector_field: np.ndarray):
    """
    direction of vector field

    Args:
        vector_field (np.ndarray): 3-D array vector field as following [w, h, c]

    Returns:
        (np.ndarray): 2-D array norm as following [w, h]

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    >>> vector_field = np.array([[[1., 1.], [2., 2.], [3., 3.]], \
                                 [[4., 4.], [5., 5.], [6., 6.]], \
                                 [[7., 7.], [8., 8.], [9., 9.]]])

    >>> direction_vector_field(vector_field=vector_field)
    array([[45., 45., 45.],
           [45., 45., 45.],
           [45., 45., 45.]])
    """

    output_width, output_height, _ = vector_field.shape
    output = np.zeros((output_width, output_height))

    for x_idx in range(output_width):
        for y_idx in range(output_height):
            if vector_field[x_idx][y_idx][0] == 0:
                vector_field[x_idx][y_idx][0] += sys.float_info.epsilon

            if vector_field[x_idx][y_idx][1] == 0:
                vector_field[x_idx][y_idx][1] += sys.float_info.epsilon

            output[x_idx][y_idx] = np.arctan(vector_field[x_idx][y_idx][0] / vector_field[x_idx][y_idx][1])

    output = (output * 180) / np.pi
    output = np.round(output, 2)
    output[output < 0] += 180

    return output


def direction_binning(direction_map: np.ndarray):
    """
    Direction Binning.
    Direction map binning to 4 direction [0, 45, 90, 135]

    Args:
        direction_map (np.ndarray): angle 2-D array map of vector field

    >>> direction_map = np.array([[45.,   2.,  70.,  90.,  153.,  170.], \
                                  [30.,  45.,  15.,  14.,   17.,  120.], \
                                  [60., 110.,  10.,  37.,   45.,   78.], \
                                  [75.,  25.,  75.,  89.,   35.,  158.], \
                                  [96.,  60.,  63., 145.,  160.,   48.]])

    >>> direction_binning(direction_map)
    array([[ 45.,   0.,  90.,  90., 135.,   0.],
           [ 45.,  45.,   0.,   0.,   0., 135.],
           [ 45.,  90.,   0.,  45.,  45.,  90.],
           [ 90.,  45.,  90.,  90.,  45.,   0.],
           [ 90.,  45.,  45., 135.,   0.,  45.]])
    """
    bin_range = {"0": [[157.5, 180], [0, 22.5]],

                 "45": [[22.5, 67.5]],

                 "90": [[67.5,112.5]],

                 "135": [[112.5, 157.5]]}

    output = copy.deepcopy(direction_map)

    keys = sorted(list(bin_range.keys()))
    for key in keys:
        for angle_range in bin_range[key]:
            min_value = angle_range[0]
            max_value = angle_range[1]
            condition = ((min_value <= output) & (output < max_value))
            output = np.where(condition, float(int(key)), output)

    return output


def non_maximum_suppression(magnitude_map: np.ndarray,
                            direction_map: np.ndarray,
                            high_threshold: int = 200,
                            low_threshold: int = 100,
                            mask_size: Tuple[int, int] = (3, 3)):
    """
    NMS(Non Maximum Suppression) Algorithms
    Refer here: https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression

    Args:
        magnitude_map (np.ndarray): magnitude 2-D array map of vector field
        direction_map (np.ndarray): angle 2-D array map of vector field
        high_treshold (int):
        low_treshold  (int):
        mask_size     (Tuple[int, int]): mask size

    Returns:
        (np.ndarray): first processed edge map

    >>> import numpy as np
    >>> np.set_printoptions(precision=2)

    >>> magnitude_map = np.array([[4.,  2.,  1.,  1.,   4.,  5.], \
                                  [5.,  7.,  3.,  6.,   2.,  9.], \
                                  [1.,  2.,  1., 10.,   4.,  5.], \
                                  [9.,  10., 4.,  3.,   2., 15.], \
                                  [19., 17., 1., 10.,   4., 20.]])

    >>> direction_map = np.array([[45.,  0., 90., 90.,135.,   0.], \
                                  [45., 45.,  0.,  0.,  0., 135.], \
                                  [45., 90.,  0., 45., 45.,  90.], \
                                  [90., 45., 90., 90., 45.,   0.], \
                                  [90., 45., 45.,135.,  0.,  45.]])
    >>> high_threshold, low_threshold = 5, 3
    >>> mask_size = (3, 3)

    >>> non_maximum_suppression(magnitude_map=magnitude_map, \
                                direction_map=direction_map, \
                                mask_size=mask_size, \
                                high_threshold=high_threshold, \
                                low_threshold=low_threshold)
    array([[0., 0., 0., 0., 1., 1.],
           [1., 1., 1., 1., 0., 1.],
           [0., 0., 0., 1., 0., 1.],
           [1., 1., 1., 0., 0., 1.],
           [1., 1., 0., 1., 0., 1.]])
    """
    mask_width, mask_height = mask_size

    assert (mask_width != 0) and (mask_height != 0), f"Mask size should be greater than 0 : {mask_size}"
    assert (mask_width % 2 == 1) and (mask_height % 2 == 1), f"Mask element should be odd : {mask_size}"

    pad_size = ((mask_width - 1) // 2, (mask_height - 1) // 2)
    pad_value = 0

    paded_magnitude_map = pad2d(source=magnitude_map, pad_size=pad_size, pad_value=pad_value)

    edge_map = np.zeros(paded_magnitude_map.shape)
    output = np.zeros(magnitude_map.shape)
    output_height, output_width = output.shape

    for y_idx in range(output_height):
        if y_idx == 5:
            break
        for x_idx in range(output_width):
            if x_idx == 6:
                break

            center_x = x_idx + pad_size[0]
            center_y = y_idx + pad_size[1]

            direction = direction_map[y_idx][x_idx]

            previous_pixel, next_pixel = get_neighbor_pixels(source=paded_magnitude_map,
                                                             point=(center_x, center_y),
                                                             direction=direction)
            center_pixel = paded_magnitude_map[center_y][center_x]


            #if y_idx == 2:
            if x_idx == 6:
                raise RuntimeError(f"{direction} {[[previous_pixel, center_pixel, next_pixel]]} {output_width}")


            previous_pixel, center_pixel, next_pixel = hysteresis_threshold(source=[previous_pixel, center_pixel, next_pixel],
                                                                            high_threshold=high_threshold,
                                                                            low_threshold=low_threshold)

            edge_map = set_neighbor_pixels(edge_map,
                                           point=(center_y, center_x),
                                           direction=direction,
                                           neighbour_pixels=[previous_pixel, center_pixel, next_pixel])

    output = edge_map[pad_size[1]:(-pad_size[1]), pad_size[0]:(-pad_size[0])]

    return output


def get_neighbor_pixels(source: np.ndarray,
                        point: Tuple[int, int],
                        direction: Union[int, float]) -> List[Union[int, float]]:
    """

    Args:
        source (np.ndarray):
        point (Tuple[int, int]):
        direction (Union[int, float]):

    Returns:
        (List[Union[int, float], Union[int, float]])
    """

    center_x, center_y = point

    # reference is vector direction as following
    # vector representation (start, end) = (previous pixel, next pixel)
    # 0:    (1, 0)
    # 45:   (1, 1)
    # 90:   (0, 1)
    # 135:  (-1, 1)
    # Arrow tip is next pixel
    # Arrow root is previous pixel
    if direction == 0.:
        next_pixel = source[center_y][center_x + 1]
        previous_pixel = source[center_y][center_x - 1]
    elif direction == 45.:
        next_pixel = source[center_y - 1][center_x + 1]
        previous_pixel = source[center_y + 1][center_x - 1]
    elif direction == 90.:
        next_pixel = source[center_y - 1][center_x]
        previous_pixel = source[center_y + 1][center_x]
    elif direction == 135.:
        next_pixel = source[center_y - 1][center_x - 1]
        previous_pixel = source[center_y + 1][center_x + 1]
    else:
        raise RuntimeError(f"not support direction value : {direction}")

    return [previous_pixel, next_pixel]


def set_neighbor_pixels(source: np.ndarray,
                        point: Tuple[int, int],
                        direction: Union[int, float],
                        neighbour_pixels: List[Union[int, float]]) -> np.ndarray:
    """

    Args:
        source              (np.ndarray):
        point               (Tuple[int, int]):
        direction           (Union[int, float]):
        neighbour_pixels    (List[Union[int, float], Union[int, float], Union[int, float]]):

    Returns:
        (np.ndarray)

    """
    previous_pixel, center_pixel, next_pixel = neighbour_pixels
    center_x, center_y = point
    output = copy.deepcopy(source)
    if direction == 0.:
        output[center_y][center_x - 1] = insert(output[center_y][center_x - 1], next_pixel)
        output[center_y][center_x + 1] = insert(output[center_y][center_x + 1], previous_pixel)
    elif direction == 45.:
        output[center_y - 1][center_x + 1] = insert(output[center_y - 1][center_x + 1], next_pixel)
        output[center_y + 1][center_x - 1] = insert(output[center_y + 1][center_x - 1], previous_pixel)
    elif direction == 90.:
        output[center_y - 1][center_x] = insert(output[center_y - 1][center_x], next_pixel)
        output[center_y + 1][center_x] = insert(output[center_y + 1][center_x], previous_pixel)
    elif direction == 135.:
        output[center_y - 1][center_x - 1] = insert(output[center_y - 1][center_x - 1], next_pixel)
        output[center_y + 1][center_x + 1] = insert(output[center_y + 1][center_x + 1], previous_pixel)

    else:
        raise RuntimeError(f"not support direction value : {direction}")

    output[center_y][center_x] = center_pixel

    return output


# TODO. Should refactoring
def insert(source, value):
    return value if source == 0 else source


def hysteresis_threshold(source: List[Union[int, float]],
                         high_threshold: int,
                         low_threshold: int) -> List[Union[int, float]]:
    """

    Args:
        source (List[Union[int, float]]): centre point pixel with neighbour pixel
        high_threshold (int): high threshold value for hysteresis threshold
        low_threshold (int): low threshold value for hysteresis threshold

    Returns:
        (np.ndarray): result of hysteresis threshold as 2-D array map


    >>> source = [6, 10, 3]
    >>> high_threshold, low_threshold = 8, 5

    >>> hysteresis_threshold(source=source, high_threshold=high_threshold, low_threshold=low_threshold)
    [1, 1, 0]
    """

    previous_pixel, center_pixel, next_pixel = source

    out_previous_pixel = 0
    out_center_pixel = 0
    out_next_pixel = 0

    if center_pixel >= high_threshold:
        out_center_pixel = 1

        if previous_pixel >= low_threshold:
            out_previous_pixel = 1

        if next_pixel >= low_threshold:
            out_next_pixel = 1

    return [out_previous_pixel, out_center_pixel, out_next_pixel]


if __name__ == "__main__":
    import os
    import doctest
    doctest.testmod()
    """
    image_dir = ""
    image_name = "Lenna.png"
    image = Image.open(os.path.join(image_dir, image_name)).convert('L')
    # Resize 14 x 14
    # This Case All Feature is Elimination
    #image = image.resize((14, 14))

    image = np.array(image)
    grayscale_image = rgb2gray(image)
    gaussian_image = gaussian_smoothing(grayscale_image,
                                        stride=1,
                                        pad_size=(0, 0),
                                        pad_value=0)

    sobel_gx = sobel_x(source=grayscale_image,
                       stride=1,
                       pad_size=(0, 0),
                       pad_value=0)

    sobel_gy = sobel_y(source=grayscale_image,
                       stride=1,
                       pad_size=(0, 0),
                       pad_value=0)

    vector_field = gradient_vector_field(sobel_gx, sobel_gy)
    magnitude_map = norm_vector_field(vector_field)
    angle_map = direction_vector_field(vector_field)
    bined_angle_map = direction_binning(angle_map)
    print(vector_field)
    print("====================================")
    print(magnitude_map)
    print("====================================")
    print(angle_map)
    print("====================================")
    print(bined_angle_map)
    minima_image = non_maximum_suppression(magnitude_map, bined_angle_map, (3, 3))

    plt.figure("NMS")
    plt.imshow(minima_image, cmap='gray')
    plt.show()

    plt.figure("Magnitude")
    plt.imshow(magnitude_map, cmap='gray')
    plt.show()
    """




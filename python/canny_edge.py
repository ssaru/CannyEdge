import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Any

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


def gradient_vector_field(gradient: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    pass


if __name__ == "__main__":
    import os
    import doctest
    doctest.testmod()

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

    gradient = (sobel_gx, sobel_gy)

    print(sobel_gx)
    print(sobel_gy)

    print(image.shape)
    print(gaussian_image.shape)
    plt.figure()
    plt.imshow(sobel_gx, cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(sobel_gy, cmap='gray')
    plt.show()




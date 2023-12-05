import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""


def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        padding_image = np.pad(image, ((4, 4), (4, 4), (0, 0)), 'constant')

        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image

        upper_left = np.random.randint(0, 8, (2), dtype=int)
        x_upper_left = upper_left[0]
        y_upper_left = upper_left[1]


        image = padding_image[x_upper_left: x_upper_left + 32, y_upper_left: y_upper_left + 32 ]


        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.

        chance = np.random.choice([True, False])
        if(chance):
            image = np.flip(image, axis = 1)

        ### YOUR CODE HERE

    ### YOUR CODE HERE
    else:
        image = image
        
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    image = (image - mean) / std

    ### YOUR CODE HERE
    
    return image

import numpy as np
  
def data_normalization_augmentation(x, y, training):
    
    if training: 
        augmented_x = []
        augmented_y = []

        for i in range(x.shape[0]):
            # Original sample
            original_sample = x[i]
            original_label = y[i]

            # Append the original sample to the augmented dataset
            augmented_x.append(original_sample)
            augmented_y.append(original_label)

            # Generate additional augmented samples
            if np.random.rand() < 0.5:
                augmented_sample = np.flipud(original_sample)
                augmented_x.append(augmented_sample)
                augmented_y.append(original_label)

        new_x = np.array(augmented_x)
        new_y = np.array(augmented_y)
    else:
        new_x = x
        new_y = y 
        
    mean = np.mean(new_x, axis=(0,1))# this compute (3,), which are means for 'open','close','volume' channel
    std = np.std(new_x, axis=(0,1))
    new_x = (new_x - mean) / std

    return new_x, new_y


def rolling_array(x, y, window_x, window_y):
    x = rolling(x[:-window_y], window=window_x)
    y = rolling(y[window_x:],window=window_y)    
    y = classification(y)
    return x, y 

def rolling(a, window=60):
    n = a.shape[0]
    return np.stack((a[i:i + window] for i in range(0,n - window + 1)),axis=0)

def classification(array):
    array = np.mean(array, axis=(1,2))
    result = np.zeros_like(array, dtype=int)  

    for i in range(array.shape[0]):
        if array[i] < -0.002:
            result[i] = 0
        elif -0.002 <= array[i] < 0.002:
            result[i] = 1
        elif array[i] >= 0.002:
            result[i] = 2
    return result

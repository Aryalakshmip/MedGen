import torch
import cv2
import numpy as np


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, attention_weights, alpha=0.5):
    """
    Generate attention heatmap overlay on the input image.
    
    Args:
        image: numpy array of shape (H, W, C) or (H, W)
        attention_weights: numpy array of attention weights
        alpha: transparency factor for heatmap overlay
    
    Returns:
        heatmap: numpy array of the same shape as image with heatmap overlay
    """
    # Ensure image is 3 channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert image to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize attention weights to match image dimensions
    attention_weights = cv2.resize(attention_weights, (image.shape[1], image.shape[0]))
    
    # Normalize attention weights to [0, 1]
    attention_min = attention_weights.min()
    attention_max = attention_weights.max()
    if attention_max > attention_min:
        attention_weights = (attention_weights - attention_min) / (attention_max - attention_min)
    else:
        attention_weights = np.zeros_like(attention_weights)
    
    # Convert to heatmap
    heatmap = np.uint8(attention_weights * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Ensure both image and heatmap have same dimensions
    assert image.shape == heatmap.shape, f"Image shape {image.shape} != Heatmap shape {heatmap.shape}"
    
    # Overlay heatmap on image
    output = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return output
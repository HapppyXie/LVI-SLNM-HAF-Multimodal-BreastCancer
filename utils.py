import numpy as np
import matplotlib.pyplot as plt

def masked_Zscore_norm(img, mask_min, mask_max, percentile_min=None, percentile_max=None):
    img_mask = np.logical_and(img > mask_min, img <= mask_max).astype(np.uint8)
    if percentile_min:
        Lower_value = np.percentile(img[img_mask == 1], percentile_min)
    else:
        Lower_value = np.min(img[img_mask == 1])
    if percentile_max:
        upper_value = np.percentile(img[img_mask == 1], percentile_max)
    else:
        upper_value = np.max(img[img_mask == 1])
    img_norm = np.clip(img, Lower_value, upper_value)
    img_mean = np.mean(img_norm[img_mask == 1])
    img_std = np.std(img_norm[img_mask == 1])
    img_norm = (img_norm - img_mean) / img_std
    img_norm *= img_mask
    return img_norm

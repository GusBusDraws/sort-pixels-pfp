from configparser import Interpolation
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


def plot_imgs(imgs, show_hist=False, fig_w=4, dpi=300, constrained_layout=True):
    if show_hist:
        nrows = 2
    else:
        nrows = 1
    if not isinstance(imgs, list):
        imgs = [imgs]
    ncols = len(imgs)
    img_w = imgs[0].shape[1]
    img_h = imgs[0].shape[0]
    xlabel_offset = 0.4
    fig_h = fig_w * (img_h / img_w) * (nrows / ncols) + xlabel_offset
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=constrained_layout
    )
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    else:
        ax = ax.ravel()
    for i, img in enumerate(imgs):
        ax[i].imshow(img, interpolation='nearest')
        ax[i].set_axis_off()
        if show_hist:
            hist, hist_centers = exposure.histogram(img)
            ax[ncols + i].plot(hist_centers, hist)
            ax[ncols + i].get_yaxis().set_visible(False)
            if img.dtype == np.uint8:
                xlabels = np.arange(0, 255, 50)
            else:
                xlabels = [round(val, 1) for val in np.arange(0, 1.1, 0.2)]
            ax[ncols + i].set_xticks(xlabels)
            ax[ncols + i].set_xticklabels(xlabels, rotation=90)
    return fig, ax

def plot_hist(img, show_img=True, fig_w=4, dpi=300, constrained_layout=True):
    nrows = 1
    if show_img:
        ncols = 2
    else:
        ncols = 1
    img_w = img.shape[1]
    img_h = img.shape[0]
    fig_h = fig_w * (img_h / img_w) * (ncols / nrows)
    fig, ax = plt.subplots(
        nrows, ncols, dpi=dpi, constrained_layout=constrained_layout
    )
    if ncols == 1:
        ax = [ax] 
    i = 0
    if show_img:
        ax[0].imshow(img, interpolation='nearest')
        ax[0].set_axis_off()
        i = 1
    hist, hist_centers = exposure.histogram(img)
    ax[i].plot(hist_centers, hist)
    return fig, ax

def sort_pixels(
    img, chans=[0, 1, 2], reverse=[False, False,False]
):
    img_sorted = img.copy()
    for chan_i in chans:
        for col_i in range(img.shape[1]):
            col = img[:, col_i, chan_i].copy()
            col = sorted(col, reverse=reverse[chan_i])
            img_sorted[:, col_i, chan_i] = col
    return img_sorted

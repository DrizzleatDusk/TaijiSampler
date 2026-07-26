import numpy as np
import matplotlib.pyplot as plt
import corner

tableau10_ = [
    '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', 
    '#59a14f', '#edc949', '#af7aa1', '#ff9da7', 
    '#9c755f', '#bab0ac'
]

def plot_corner(data, bins=40, interval=None, axes_scale="linear",
                weights=None, color='#4e79a7', smooth=.9, smooth1d=None,
                labels=None, label_kwargs=None, titles=None, show_titles=False,
                title_quantiles=None, title_fmt=".2f", title_kwargs=None,
                truths=None, truth_color="#e15759", quantiles=None, fig=None,
                max_n_ticks=4, hist_kwargs=None, **hist2d_kwargs):
    if quantiles is None:
        quantiles = (.16, .5, .84)
    hist2d_kwargs["levels"] = hist2d_kwargs.get("levels", (.68, .95))
    hist2d_kwargs["plot_datapoints"] = hist2d_kwargs.get("plot_datapoints", False)
    hist2d_kwargs["fill_contours"] = hist2d_kwargs.get("fill_contours", True)

    newfig = corner.corner(data, bins, range=interval, axes_scale=axes_scale,
                    weight=weights, color=color, smooth=smooth, smooth1d=smooth1d,
                    labels=labels, label_kwargs=label_kwargs, titles=titles, show_titles=show_titles,
                    title_quantiles=title_quantiles, title_fmt=title_fmt, title_kwargs=title_kwargs,
                    truths=truths, truth_color=truth_color, quantiles=quantiles, fig=fig,
                    max_n_ticks=max_n_ticks, hist_kwargs=hist_kwargs, **hist2d_kwargs,
                    )
    return newfig
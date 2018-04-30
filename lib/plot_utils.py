import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO


def traffic_density_plot(lat=[], lon=[], directory=None, length_cutoff=600, subfix=""):
    """
    Visualize density of geometric traffic

    Args:
        lat (list[float]):
        lon (list[float]):
        directory (str):
        length_cutoff (int):
        subfix (str):

    Returns:

    """

    xmin = min(lon)
    xmax = max(lon)
    ymin = min(lat)
    ymax = max(lat)

    plt.figure(figsize=(20, 10))
    fig = plt.figure(frameon=False) 
    fig.set_size_inches(20,20)

    # To make the content fill the whole figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()

    # Set a single t value to slice the multidimensional array.
    length_cutoff = length_cutoff-100
    # With the above note in mind, this may be an exception??
    x1=np.copy(lon)
    y1=np.copy(lat)

    # Remove the nans from the array
    x1 = x1[~np.isnan(x1)]
    y1 = y1[~np.isnan(y1)]

    # Log colormap
    hb = ax.hexbin(
        x1,
        y1,
        gridsize=500,
        bins='log',
        cmap='inferno',
        extent=(xmin, xmax, ymin, ymax)
    )

    fig.add_axes(ax)
    ax.axis('equal')
    plt.axis('off')
    # Setting the axes like this avoid the zero values in
    # the preallocated empty array.
    ax.axis([xmin, xmax, ymin, ymax])

    if not directory:
        plt.show()
        return 1
    else:
        # save figure as png

        if not os.path.exists(directory):
            os.makedirs(directory)
        # png1 = BytesIO()
        filename= "{}/{}_traffic_density.png".format(directory.rstrip("/"), subfix)
        fig.savefig(filename, format='png', bbox_inches='tight', pad_inches=0)
        return 1

def traffic_flight_plot():
    # TODO: implement the visualization of clustering result
    pass
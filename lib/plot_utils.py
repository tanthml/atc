import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt


def traffic_density_plot(lat=[], lon=[], file_path=None, length_cutoff=600):
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

    if not file_path:
        plt.show()
        return 1
    else:
        # # save figure as png
        #
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # # png1 = BytesIO()
        fig.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)
        return 1


def traffic_flight_plot(flight_ids, clusters, unique_labels, flight_dicts, file_path):
    """
    visualization of clustering result
    Args:
        flight_ids:
        clusters:
        unique_labels:
        flight_dicts:
        file_path:

    Returns:

    """
    # TODO: implement the visualization of clustering result

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(set(unique_labels)))]

    plt.figure(figsize=(20, 10))
    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 20)
    ax = fig.add_subplot(1, 1, 1)
    # And a corresponding grid
    ax.grid(which='both')
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.9)

    for index, code in enumerate(flight_ids):
        if clusters[index] == -1:
            # logger.info("outlier")
            continue
        x = flight_dicts[code][:, 1]  # lon
        y = flight_dicts[code][:, 0]  # lat
        label = clusters[index]
        color = colors[label]
        ax.scatter(
            x=x,  # x axis
            y=y,  # y axis
            alpha=0.9,
            label=label,
            color=color,
            cmap=plt.cm.jet,
            s=1,
        )
    # export images
    plt.savefig(
        "../tmp/{file_path}".format(
            file_path=file_path
        )
    )

def plot_centermost_points(reduced_groups):
    pass
    # grid group reduce size
    # print(reduced_groups.head())
    # centermost_points = reduced_groups.map(get_centermost_point)
    # lats, lons = zip(*centermost_points)
    # ax.scatter(
    #     x=lons,  # x axis
    #     y=lats,  # y axis
    #     cmap=plt.cm.jet,
    #     c='#99cc99', edgecolor='None', alpha=0.7, s=160
    # )

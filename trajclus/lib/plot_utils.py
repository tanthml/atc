import numpy as np
# matplotlib.use('Agg')
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


def traffic_density_plot(lat=[], lon=[], file_path=None, length_cutoff=600):
    """
    Visualize density of geometric traffic

    Args:
        lat (list[float]):
        lon (list[float]):
        length_cutoff (int):

    Returns:

    """

    xmin = min(lon)
    xmax = max(lon)
    ymin = min(lat)
    ymax = max(lat)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 20)
    # To make the content fill the whole figure
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax = plt.Axes(fig, [0.05, 0.05, 0.95, 0.95])
    # fig.add_axes(ax)
    ax = fig.add_subplot(1, 1, 1)

    # And a corresponding grid
    # ax.grid(which='both')

    # Set a single t value to slice the multidimensional array.
    length_cutoff = length_cutoff-100
    # With the above note in mind, this may be an exception??
    x1=np.copy(lon)
    y1=np.copy(lat)

    # Remove the nans from the array
    x1 = x1[~np.isnan(x1)]
    y1 = y1[~np.isnan(y1)]
    # plt.title(file_path.split("/")[-1].split(".")[0], fontsize=30)
    plt.xlabel('Longitude', fontsize=24)
    plt.ylabel('Latitude', fontsize=24)
    # Log colormap
    hb = ax.hexbin(
        x1,
        y1,
        gridsize=1000,
        bins='log',
        cmap='inferno',
        extent=(xmin, xmax, ymin, ymax)
    )

    fig.add_axes(ax)
    # ax.axis('equal')
    plt.axis('on')
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
        fig.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        return 1


def traffic_flight_plot(
        flight_ids, clusters, flight_dicts, file_path, group_clusters, info={}):
    """
    visualization of clustering result
    Args:
        flight_ids (list[str]):
        clusters (list[]):
        flight_dicts: (dict)
        group_clusters list[Any]:
        file_path (str):

    Returns:

    """
    # TODO: implement the visualization of clustering result

    unique_labels = set(clusters)

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(set(unique_labels)))]

    colors_dict = {}
    for idx, uni in enumerate(unique_labels):
        colors_dict[uni] = colors[idx]

    # plt.style.use('dark_background')
    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 20)
    # To make the content fill the whole figure
    ax = plt.Axes(fig, [0.05, 0.05, 0.95, 0.95])
    # fig.add_axes(ax)
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_facecolor("grey")

    # And a corresponding grid
    ax.grid(False)
    # Or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.2)
    # ax.grid(which='major', alpha=0.5)

    for index, code in enumerate(flight_ids):
        if clusters[index] == 0:
            # logger.info("outlier")
            continue
        x = flight_dicts[code][:, 1]  # lon
        y = flight_dicts[code][:, 0]  # lat
        label = clusters[index]
        color = colors_dict[label]
        plt.title("{} {}".format(info['airport_code'], info['file_name']),
                  fontsize=24)
        plt.xlabel('Longitude', fontsize=24)
        plt.ylabel('Latitude', fontsize=24)
        plt.plot(x, y, '-ok', color=color,
                 markersize=0, linewidth=1,
                 markerfacecolor='white',
                 markeredgecolor='gray',
                 markeredgewidth=1)
    plt.legend()

    plt.savefig(
        "../tmp/{file_path}".format(
            file_path=file_path
        ),
        dpi=300
    )

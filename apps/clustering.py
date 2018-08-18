from time import gmtime, strftime

import click
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from lib.common_utils import gen_log_file
logger = gen_log_file(path_to_file='tmp/clustering.log')
from lib.plot_utils import traffic_density_plot, traffic_flight_plot
from lib.geometric_utils import build_coordinator_dict, flight_id_encoder,\
    build_matrix_distances


def cluster_trajectories(dist_matrix, epsilon=1, min_samples=1):
    """
    Building cluster from distance matrix of all flight

    Args:
        dist_matrix ():
        epsilon (float):
        min_samples (int):

    Returns:
        clusters ()
        labels (list[int]): list of cluster id

    """

    db = DBSCAN(
        eps=epsilon,
        min_samples=min_samples,
        algorithm='auto',
        metric='precomputed'
    )
    db.fit(X=dist_matrix)

    labels = db.labels_
    num_clusters = len(set(labels))
    clusters = pd.Series(
        [dist_matrix[labels == idx] for idx in range(num_clusters)]
    )
    silhouette_val = silhouette_score(
        X=dist_matrix,
        labels=labels,
        metric='precomputed'
    )

    logger.info(
        'Number of clusters: {} with Silhouette Coefficient {}'.format(
            num_clusters, silhouette_val
        )
    )
    return clusters, labels, silhouette_val


@click.command()
@click.option(
    '--input_path',
    type=str,
    required=True,
    help='Full path to the trajectory file in CSV format')
@click.option(
    '--airport_code',
    type=str,
    default='WSSS',
    help='Air Port Codename')
@click.option(
    '--distance',
    type=str,
    default='directed_hausdorff',
    help='Distance algorithm current support: directed_hausdorff, frechet')
@click.option(
    '--min_sample',
    type=int,
    default=4,
    help='Min sample value in DBSCAN')
def main(input_path, airport_code, distance, min_sample):
    history = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "_")
    logger.info("=============================================")
    logger.info("================ DATETIME {} ================".format(history))
    df = pd.read_csv(input_path)
    logger.info(df.head())
    file_name = input_path.split("/")[-1].replace(".csv", "")

    departure_airports = df['Origin'].unique()
    destination_airports = df['Destination'].unique()
    one_airport = df[(df['Destination'] == airport_code)]

    # get fixed
    flights_toward_airport = one_airport[(one_airport['DRemains'] < 2.0)
                                         & (one_airport['DRemains'] > 0.01)]
    traffic_density_plot(
        lat=flights_toward_airport['Latitude'],
        lon=flights_toward_airport['Longitude'],
        directory="tmp",
        length_cutoff=600,
        subfix="{}_{}".format(file_name, airport_code)
    )

    logger.info("Encoding flight ID ...")
    flight_ids = flights_toward_airport['Flight_ID'].unique().tolist()
    logger.info("Total # flight ID {}".format(len(flight_ids)))
    flight_encoder = flight_id_encoder(flight_ids)

    logger.info("Extracting coordinate and flight id from dataset")
    encoded_idx, coord_list, flight_dicts,  = build_coordinator_dict(
        df=flights_toward_airport,
        label_encoder=flight_encoder,
        flight_ids=flight_ids,
        max_flights=1000
    )

    # create dataframe result
    clusters_df = pd.DataFrame()
    clusters_df['Flight_ID'] = flight_encoder.inverse_transform(encoded_idx)

    logger.info("Building distance matrix - {} ...".format(distance))
    dist_matrix = build_matrix_distances(coord_list, dist_type=distance)

    # prepare grid search for tuning eps
    alpha = 0.01
    upper_bound = max(dist_matrix[0,:])
    lower_bound = min(dist_matrix[0,:])
    step = (upper_bound - lower_bound) * alpha
    logger.info(
        "upper_bound {}, lower_bound {}, step {}".format(
            upper_bound, lower_bound, step)
    )

    kms_per_radian = 6371.0088
    last_clusters = None
    for eps in np.arange(step*2, step*5, step):
        epsilon = eps
        # epsilon =  eps / kms_per_radian
        clusters, labels, silhouette = cluster_trajectories(
            dist_matrix=dist_matrix,
            epsilon=epsilon,
            min_samples=min_sample
        )

        # list of cluster id along side with the  encoded flight id
        last_clusters = clusters
        unique_labels = set(labels)
        clusters_df['c_{}_eps_{}'.format(len(unique_labels), epsilon)] = labels

        logger.info(unique_labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        plt.figure(figsize=(20, 10))
        fig = plt.figure(frameon=False)
        fig.set_size_inches(20, 20)
        ax = fig.add_subplot(1, 1, 1)
        # And a corresponding grid
        ax.grid(which='both')
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        for index, code in enumerate(encoded_idx):
            if labels[index] == -1:
                # logger.info("outlier")
                continue
            x = flight_dicts[code][:, 1]  # lon
            y = flight_dicts[code][:, 0]  # lat
            label = labels[index]
            color = colors[label]
            ax.scatter(
                x=x,  # x axis
                y=y,  # y axis
                alpha=0.9,
                label=label,
                color=color,
                cmap=plt.cm.jet
            )
        # export images
        plt.savefig(
            "tmp/{}_{}_ms_{}_eps_{}_sil_{}.png".format(
                file_name, airport_code, min_sample, epsilon, silhouette
            )
        )
        if len(last_clusters) <= 2:
            break
    clusters_df.to_csv(
        "tmp/{}_{}_ms_{}.csv".format(
            file_name, airport_code, min_sample
        ),
        index=False
    )
    logger.info("\n {}".format(clusters_df.head()))


if __name__ == '__main__':
    main()

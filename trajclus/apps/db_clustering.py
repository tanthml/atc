from time import gmtime, strftime

import click
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from trajclus.lib.common_utils import gen_log_file
from trajclus.lib.preprocessing_lib import filter_by_airport, flight_id_encoder, \
    build_flight_trajectory_df
from trajclus.lib.plot_utils import traffic_density_plot, traffic_flight_plot
from trajclus.lib.geometric_utils import build_matrix_distances


logger = gen_log_file(path_to_file='../tmp/db_clustering.log')


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
def main(input_path, airport_code, distance, min_sample, max_flights=1000):
    history = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "_")
    logger.info("=============================================")
    logger.info("================ DATETIME {} ================".format(history))
    df = pd.read_csv(input_path)
    logger.info(df.head())
    file_name = input_path.split("/")[-1].replace(".csv", "")

    # get fixed
    flights_to_airport = filter_by_airport(
        df=df,
        airport_code=airport_code,
        min_dr=0.01,
        max_dr=2.0
    )
    file_path = "../tmp/{file_name}_{airport_code}_traffic_density.png".format(
        file_name=file_name,
        airport_code=airport_code
    )
    traffic_density_plot(
        lat=flights_to_airport['Latitude'],
        lon=flights_to_airport['Longitude'],
        file_path=file_path,
        length_cutoff=600
    )

    logger.info("Encoding flight ID ...")
    flight_ids = flights_to_airport['Flight_ID'].unique().tolist()
    logger.info("Total # flight ID {}".format(len(flight_ids)))
    flight_encoder = flight_id_encoder(flight_ids)

    logger.info("Extracting trajectory coordinators and flight id from dataset")
    flight_df, flight_dicts = build_flight_trajectory_df(
        flights_to_airport=flights_to_airport,
        label_encoder=flight_encoder,
        flight_ids=flight_ids,
        max_flights=max_flights,
        is_simplify=True
    )

    # create data-frame result
    clusters_df = pd.DataFrame()
    clusters_df['Flight_ID'] = flight_encoder.inverse_transform(flight_df['idx'])

    logger.info("Building distance matrix - {} ...".format(distance))
    dist_matrix = build_matrix_distances(
        coords=flight_df['trajectory'],
        dist_type=distance
    )

    # prepare grid search for tuning epsilon
    alpha = 0.01
    upper_bound = max(dist_matrix[0,:])
    lower_bound = min(dist_matrix[0,:])
    step = (upper_bound - lower_bound) * alpha
    logger.info(
        "upper_bound {}, lower_bound {}, step {}".format(
            upper_bound, lower_bound, step)
    )

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

        # export images
        result_file_name = "../tmp/{}_{}_dbscan_ms_{}_eps_{}_sil_{}.png".format(
                file_name, airport_code, min_sample, epsilon, silhouette
            )
        traffic_flight_plot(
            flight_ids=flight_df['idx'].tolist(),
            clusters=labels,
            unique_labels=unique_labels,
            flight_dicts=flight_dicts,
            file_path=result_file_name
        )
        if len(last_clusters) <= 2:
            break

    # export result
    clusters_df.to_csv(
        "../tmp/{}_{}_ms_{}.csv".format(
            file_name, airport_code, min_sample
        ),
        index=False
    )
    logger.info("\n {}".format(clusters_df.head()))


if __name__ == '__main__':
    main()

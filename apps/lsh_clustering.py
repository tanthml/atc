import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn.metrics import silhouette_score

from lib.common_utils import gen_log_file
from lib.preprocessing_lib import filter_by_airport, \
    build_flight_trajectory_df, flight_id_encoder
from lib.geometric_utils import KM_PER_RADIAN, simplify_coordinator
from lib.lsh_lib import LSHClusteringLib
from lib.plot_utils import traffic_flight_plot


logger = gen_log_file(path_to_file='../tmp/lsh_clustering.log')


def dbscan_clustering(coords, min_sample=1, max_distance=1.0, epsilon=None):
    """
    Mapping all points in map to reduced cluster
    Args:
        coords :
        min_sample (int):
        max_distance (float):
        epsilon (float):

    Returns:

    """

    """
    The epsilon parameter is the max distance (max_distance) 
    that points can be from each other to be considered a cluster.
    """
    if not epsilon:
        epsilon = max_distance / KM_PER_RADIAN
    db = DBSCAN(eps=epsilon, min_samples=min_sample, algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series(
        [coords[cluster_labels == n] for n in range(num_clusters)])
    print('Number of clusters for grouping: {}'.format(num_clusters))
    return clusters, db


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def compute_silhouette_score(feature_matrix, labels):
    silhouette_val = silhouette_score(
        X=feature_matrix,
        labels=labels,
        metric='haversine'
    )
    return silhouette_val


def main(input_path, airport_code='WSSS', max_flights=1000):
    # load raw-data from csv
    df = pd.read_csv(input_path)
    file_name = input_path.split("/")[-1].replace(".csv", "")

    # filter data by airport code-name
    flights_to_airport = filter_by_airport(
        df=df,
        airport_code=airport_code,
        min_dr=0.01,
        max_dr=2.0
    )
    print(flights_to_airport[['DRemains', 'Latitude', 'Longitude']].head())

    logger.info("Encoding flight ID ...")
    flight_ids = flights_to_airport['Flight_ID'].unique().tolist()
    logger.info("Total # flight ID {}".format(len(flight_ids)))
    flight_encoder = flight_id_encoder(flight_ids)

    flight_df, flight_dicts = build_flight_trajectory_df(
        flights_to_airport=flights_to_airport,
        label_encoder=flight_encoder,
        flight_ids=flight_ids,
        max_flights=max_flights,
        is_simplify=True
    )

    simplified_coords = [simplify_coordinator(coord_curve=curve, epsilon=0.001)
                         for curve in flight_df['trajectory']
                         ]
    coords = simplified_coords[0]
    for item in simplified_coords[1:]:
        coords = np.concatenate((coords, item))
    print("Total points %s" % len(coords))

    reduced_groups, classifier = dbscan_clustering(
        coords=coords,
        min_sample=1,
        max_distance=1.5
    )
    # we trick each group label as a term, then each trajectory will contains
    # list of terms/tokens
    flight_df['groups'] = [classifier.fit_predict(X=coord)
                           for coord in flight_df['trajectory'].tolist()]
    print(flight_df.head())
    # convert clustering number to group label,
    flight_df['groups'] = flight_df['groups'].apply(
        lambda clusters: ["G{}".format(c) for c in clusters])
    print(flight_df.head())

    # Now we will apply Jaccard similarity and LSH for theses trajectories
    lsh_clustering = LSHClusteringLib(
        threshold=0.8,
        num_perm=128
    )
    flight_df['hash'] = lsh_clustering.compute_min_hash_lsh_over_data(
        record_ids=flight_df['idx'].tolist(),
        data=flight_df['groups'].tolist()
    )

    flight_df['duplicated'] = flight_df['hash'].apply(
        lambda x: lsh_clustering.query_duplicated_record(x)
    )
    print(flight_df.head())

    flight_df['buckets'] = flight_df['duplicated'].apply(
        lambda x: '_'.join(x)
    )
    print(flight_df.head())
    unique_buckets = flight_df['buckets'].unique().tolist()
    print("number buckets %s" % len(unique_buckets))
    print(unique_buckets)
    print(len(flight_df.groupby('buckets').size()))
    n_curve_per_bucket = flight_df.groupby('buckets').size().to_dict()

    def convert_to_cluster_number(bucket_label, unique_buckets, n_curve_per_bucket=None):
        if n_curve_per_bucket[bucket_label] <= 5:
            return -1
        return unique_buckets.index(bucket_label)

    cluster_labels = [
        convert_to_cluster_number(bucket, unique_buckets, n_curve_per_bucket)
        for bucket in flight_df['buckets'].tolist()
    ]
    flight_df['cluster'] = cluster_labels
    print(flight_df.head())

    # # evaluation
    # dist_matrix = build_matrix_distances(
    #     coord_list,
    #     dist_type='directed_hausdorff'
    # )
    # silhouette_val = compute_silhouette_score(
    #     feature_matrix=dist_matrix, labels=cluster_labels
    # )

    result_file_name =  "{file_name}_{airport_code}_lsh_sil_{subfix}.png".format(
            file_name=file_name,
            airport_code=airport_code,
            subfix=None
        )
    traffic_flight_plot(
        flight_ids=flight_df['idx'].tolist(),
        clusters=cluster_labels,
        unique_labels=unique_buckets,
        flight_dicts=flight_dicts,
        file_path=result_file_name
    )


if __name__ == '__main__':
    # main("/Users/tanthm/jvn_data/NTU/tracks_2015_06_01-001.csv")
    main("/Users/tanthm/jvn_data/NTU/tracks_2016_09_01_destination_wsss.csv")

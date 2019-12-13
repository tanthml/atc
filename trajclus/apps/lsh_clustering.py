import click
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from trajclus.lib.common_utils import gen_log_file
from trajclus.lib.preprocessing_lib import filter_by_airport, \
    build_flight_trajectory_df, flight_id_encoder
from trajclus.lib.geometric_utils import KM_PER_RADIAN, simplify_coordinator, \
    build_matrix_distances
from trajclus.lib.lsh_lib import LSHClusteringLib
from trajclus.lib.plot_utils import traffic_flight_plot


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
    clusters = pd.Series([coords[cluster_labels == n]
                        for n in range(num_clusters)])
    centers = clusters.map(get_centermost_point)
    centers = np.array(centers.tolist())
    print('Number of clusters for grouping: {}'.format(num_clusters))
    return centers, db


def kmeans_clustering(coords, k_cluster):
    kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(X=coords)
    cluster_labels = kmeans.labels_
    num_clusters = len(set(cluster_labels))

    centers = np.array(kmeans.cluster_centers_)

    print('Number of clusters for grouping: {}'.format(num_clusters))
    return centers, kmeans


def get_centermost_point(cluster):
    centroid = [MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y]
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)

    return centermost_point


def compute_silhouette_score(feature_matrix, labels):
    silhouette_val = silhouette_score(
        X=feature_matrix,
        labels=labels,
        metric='precomputed'
    )
    return silhouette_val


def detect_entrance_ways(point_coords, algorithm='k-means', estimated_n_entrance=9):
    if algorithm not in ['k-means', 'dbscan']:
        return [], False
    # auto detect entrance ways
    if algorithm == 'k-means':
        return kmeans_clustering(
            coords=point_coords,
            k_cluster=estimated_n_entrance
        )
    if algorithm == 'dbscan':
        return dbscan_clustering(
            coords=point_coords,
            min_sample=1,  # must be 1
            max_distance=15.0
        )


def filter_by_date(datetime, filter_date):
    """
    Filter date
    Args:
        datetime (str): yyyy-mm-dd format
        filter_date (str): yyyy-mm-dd format

    Returns:
        (bool)
    """
    str_date = str(datetime).split(' ')[0]
    if str_date == str(filter_date):
        return True
    return False


def main(
        input_path,
        airport_code='WSSS',
        max_flights=1000,
        estimated_n_entrance=9,
        threshold=0.6,
        algo='k-means',
        min_dr=1.0,
        max_dr=2.0,
        filter_date='',
        epsilon=0.001
):
    # load raw-data from csv
    logger = gen_log_file(path_to_file='../tmp/lsh_clustering_{}.log'.format(filter_date))
    df = pd.read_csv(input_path)
    file_name = input_path.split("/")[-1].replace(".csv", "")

    if filter_date != '':
        print("before filtering %s" % len(df))
        df['filtered'] = df['Actual_Arrival_Time_(UTC)'].apply(
            lambda x: filter_by_date(datetime=x, filter_date=filter_date)
        )
        df = df[df['filtered']]
        file_name = filter_date
        print("after filtering %s" % len(df))

    # filter data by airport code-name
    flights_to_airport = filter_by_airport(
        df=df,
        airport_code=airport_code,
        min_dr=0.0,
        max_dr=max_dr
    )

    # prepare data-frame for detect entrance points toward the airport
    entrance_to_airport = filter_by_airport(
        df=df,
        airport_code=airport_code,
        min_dr=min_dr,
        max_dr=max_dr
    )

    logger.info("Encoding flight ID ... %s" % airport_code)
    flight_ids = flights_to_airport['Flight_ID'].unique().tolist()
    logger.info("Total # flight ID {}".format(len(flight_ids)))
    flight_encoder = flight_id_encoder(flight_ids)

    flight_df, flight_dicts = build_flight_trajectory_df(
        flights_to_airport=flights_to_airport,
        label_encoder=flight_encoder,
        flight_ids=flight_ids,
        max_flights=max_flights,
        epsilon=epsilon
    )

    entrance_trajectories = []
    total_original_points = 0
    for fid in flight_ids[:max_flights]:
        tmp_df = entrance_to_airport[entrance_to_airport['Flight_ID'] == fid]
        tmp_df = tmp_df.sort_values(by='DRemains', ascending=False)
        lat_lon_values = tmp_df[['Latitude', 'Longitude']].values
        total_original_points += len(lat_lon_values)
        entrance_trajectories.append(lat_lon_values)

    simplified_coords = [
        simplify_coordinator(coord_curve=curve, epsilon=epsilon)
        for curve in entrance_trajectories
    ]

    logger.info("Total original points at entrance %s" % total_original_points)
    point_coords = simplified_coords[0]
    for item in simplified_coords[1:]:
        point_coords = np.concatenate((point_coords, item))
    logger.info("Total points at entrance %s" % len(point_coords))

    detect_entrance_algo = algo
    reduced_groups, classifier = detect_entrance_ways(
        point_coords=point_coords,
        algorithm=detect_entrance_algo,
        estimated_n_entrance=estimated_n_entrance
    )

    # we trick each group label as a term, then each trajectory will contains
    # list of terms/tokens
    if detect_entrance_algo == 'dbscan':
        flight_df['groups'] = [classifier.fit_predict(X=coord)
                               for coord in entrance_trajectories]
    elif detect_entrance_algo == 'k-means':
        entrance_groups = []
        for traj in entrance_trajectories:
            if len(traj) > 1:
                entrance_groups.append(classifier.predict(X=traj))
            else:
                entrance_groups.append([-1])
        flight_df['groups'] = entrance_groups

    # convert clustering number to group label,
    flight_df['groups'] = flight_df['groups'].apply(
        lambda clusters: ["G{}".format(c) for c in clusters])

    # Now we will apply Jaccard similarity and LSH for theses trajectories
    lsh_clustering = LSHClusteringLib(
        threshold=threshold,
        num_perm=128
    )
    flight_df['hash'] = lsh_clustering.compute_min_hash_lsh_over_data(
        record_ids=flight_df['idx'].tolist(),
        data=flight_df['groups'].tolist()
    )

    flight_df['duplicated'] = flight_df['hash'].apply(
        lambda x: lsh_clustering.query_duplicated_record(x)
    )

    flight_df['buckets'] = flight_df['duplicated'].apply(
        lambda x: '_'.join(x)
    )
    unique_buckets = flight_df['buckets'].unique().tolist()
    logger.info("number buckets %s" % len(unique_buckets))
    logger.info(len(flight_df.groupby('buckets').size()))
    n_curve_per_bucket = flight_df.groupby('buckets').size().to_dict()

    def convert_to_cluster_number(
            bucket_label, unique_buckets, total_buckets, n_curve_per_bucket=None):
        # less number in bucket will be consider as outliers , label = -1
        if (n_curve_per_bucket[bucket_label] * 100.0 / total_buckets) <= 5.0:
            return -1
        return unique_buckets.index(bucket_label)

    cluster_labels = [
        convert_to_cluster_number(bucket, unique_buckets, len(flight_df), n_curve_per_bucket)
        for bucket in flight_df['buckets'].tolist()
    ]
    flight_df['cluster'] = cluster_labels
    logger.info("Non-outlier cluster number %s" %
          len(flight_df[flight_df['cluster'] != -1]['cluster'].unique().tolist())
    )
    logger.info(flight_df[flight_df['cluster'] != -1]['cluster'].unique())
    n_curve_per_cluster = flight_df.groupby('cluster').size()
    logger.info(n_curve_per_cluster)


    # # evaluation
    silhouette_val = None
    dist_matrix = build_matrix_distances(
        coords=flight_df['trajectory'].tolist(),
        dist_type='directed_hausdorff'
    )
    silhouette_val = compute_silhouette_score(
        feature_matrix=dist_matrix, labels=cluster_labels
    )
    logger.info("Silhouette Coefficient via LSH %s" % silhouette_val)

    # ### base-line with DBSCAN
    # from db_clustering import cluster_trajectories
    # alpha = 0.001
    # upper_bound = max(dist_matrix[0, :])
    # lower_bound = min(dist_matrix[0, :])
    # step = (upper_bound - lower_bound) * alpha
    # logger.info(
    #     "upper_bound {}, lower_bound {}, step {}".format(
    #         upper_bound, lower_bound, step)
    # )
    # eps_list = np.arange(step*1, step*5, step)
    # for eps in eps_list:
    #     try:
    #         clusters, labels, silhouette = cluster_trajectories(
    #             dist_matrix=dist_matrix,
    #             epsilon=eps,
    #             min_samples=1
    #         )
    #     except:
    #         continue

    plot_file_name = "{file_name}_{airport_code}_lsh_{threshold}_{algo}_{n_entrance}_dr_{dr_range}_sil_{silhoette}.png".format(
            file_name=file_name,
            airport_code="{}_{}_flights".format(airport_code, len(flight_df)),
            threshold=threshold,
            algo=detect_entrance_algo,
            n_entrance=estimated_n_entrance,
            dr_range="{}_{}".format(min_dr, max_dr),
            silhoette = silhouette_val

        )

    traffic_flight_plot(
        flight_ids=flight_df['idx'].tolist(),
        clusters=cluster_labels,
        flight_dicts=flight_dicts,
        file_path=plot_file_name,
        group_clusters=reduced_groups,
        info={'file_name': file_name, 'airport_code': airport_code}
    )

    result_file_name = "{file_name}_{airport_code}_lsh_{threshold}_{algo}_{n_entrance}_dr_{dr_range}_sil_{silhoette}.png".format(
            file_name=file_name,
            airport_code="{}_{}_flights".format(airport_code, len(flight_df)),
            threshold=threshold,
            algo=detect_entrance_algo,
            n_entrance=estimated_n_entrance,
            dr_range="{}_{}".format(min_dr, max_dr),
            silhoette=silhouette_val

        )
    # export flight id with label of clusters to csv file
    flight_df[
        ['flight_id', 'buckets', 'cluster']
    ].to_csv("../tmp/{}.csv".format(result_file_name), index=False)


@click.command()
@click.option(
    '--input_path',
    type=str,
    required=True,
    help='Full path to the trajectory file in CSV format')
@click.option(
    '--airport_code',
    type=str,
    # default='WSSS,VTBS,WMKK',
    default='WSSS',
    help='Air Port Codename')
@click.option(
    '--max_flights',
    type=int,
    default=1000,
    help='Max number of flights')
@click.option(
    '--dr_range',
    type=str,
    default='1.0,5.0',
    help='distance remains in radius')
@click.option(
    '--filter_date',
    type=str,
    default='',
    help='Filter by date example 2016-09-29')
@click.option(
    '--epsilon',
    type=float,
    default=0.001,
    help='epsilon for simplify curve using Douglas Peucker')
def main_cli(input_path, airport_code, max_flights, dr_range, filter_date, epsilon):
    airports = airport_code.split(",")
    dr_ranges = [float(i) for i in dr_range.split(",")]
    for airport in airports:
        main(
            input_path=input_path,
            airport_code=airport,
            max_flights=max_flights,
            estimated_n_entrance=30,
            threshold=0.5,
            algo='k-means',
            min_dr=dr_ranges[0],
            max_dr=dr_ranges[1],
            filter_date=filter_date,
            epsilon=epsilon
        )


if __name__ == '__main__':
    main_cli()

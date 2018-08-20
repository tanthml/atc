import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from simplification.cutil import simplify_coords

from lib.geometric_utils import build_coordinator_dict, flight_id_encoder
from lib.lsh_lib import ClusteringLSHLib
from lib.common_utils import gen_log_file
logger = gen_log_file(path_to_file='../tmp/feature_engineering_lib.log')


KM_PER_RADIAN = 6371.0088


def dbscan_clustering(coords, min_sample=1, max_distance=1.0):
    """

    :param data:
    :param epsilon:
    :param min_sample:
    :return:
    """

    """
    The epsilon parameter is the max distance (max_distance) 
    that points can be from each other to be considered a cluster.
    """
    epsilon = None
    if not epsilon:
        epsilon = max_distance / KM_PER_RADIAN
    db = DBSCAN(eps=epsilon, min_samples=min_sample, algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series(
        [coords[cluster_labels == n] for n in range(num_clusters)])
    print('Number of clusters: {}'.format(num_clusters))
    return clusters, db

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)


def filter_data_by_airport(df, airport_code):
    one_airport = df[(df['Destination'] == airport_code)]

    # get fixed
    flights_toward_airport = one_airport[
        (one_airport['DRemains'] < 2.0) & (one_airport['DRemains'] > 1.1)]

    return flights_toward_airport

def simplify_coordinator(coord_curve, epsilon=0.001):
    """

    Args:
        coord_curve (list[list[float, float]]): a list of lat, lon coordinates
        epsilon (float):
    Returns:
        list[list[float, float]]
    """
    coord_curve = np.asarray(coord_curve, order='C')
    return simplify_coords(coord_curve, epsilon)

def main(input_path, airport_code='WSSS', lat_label='Latitude', lon_label='Longitude', sample=100):
    df = pd.read_csv(input_path)
    flights_toward_airport = filter_data_by_airport(df=df, airport_code=airport_code)
    # coords = df.as_matrix(columns=[lat_label, lon_label])
    # coords = simplify_coordinator(coords_cur=coords)
    print(flights_toward_airport.head())
    logger.info("Encoding flight ID ...")
    flight_ids = flights_toward_airport['Flight_ID'].unique().tolist()
    logger.info("Total # flight ID {}".format(len(flight_ids)))
    flight_encoder = flight_id_encoder(flight_ids)

    encoded_idx, coord_list, flight_dicts,  = build_coordinator_dict(
        df=flights_toward_airport,
        label_encoder=flight_encoder,
        flight_ids=flight_ids,
        max_flights=1000,
        is_simplify=False
    )
    flight_df = pd.DataFrame()
    flight_df['idx'] = encoded_idx
    flight_df['flight_id'] = flight_ids
    flight_df['coordinators'] = coord_list
    print(type(coord_list[0]))
    # exit()
    print(flight_df.head())

    coords = coord_list[0]
    for item in coord_list[1::]:
        coords =np.concatenate((coords, item))
    print("Total points %s" % len(coords))

    clusters, classifier = dbscan_clustering(
        coords=coords,
        min_sample=1,
        max_distance=2.0)
    centermost_points = clusters.map(get_centermost_point)

    flight_df['groups'] = [classifier.fit_predict(X=coord)
                           for coord in coord_list]
    # convert clustering number to group label,
    flight_df['groups'] = flight_df['groups'].apply(
        lambda clusters: ["G{}".format(c) for c in clusters])
    print(flight_df.head())
    # we trick each group label as a term, then each trajectory will contains
    # list of terms/tokens

    # Now we will apply Jaccard similarity and LSH for theses trajectories
    lsh_clustering = ClusteringLSHLib(
        threshold=0.8
    )
    flight_df['hash'] = lsh_clustering.compute_min_hash_lsh_over_data(
        record_ids=flight_df['idx'].tolist(),
        data=flight_df['groups'].tolist()
    )

    flight_df['duplicated'] = flight_df['hash'].apply(
        lambda x: lsh_clustering.query_duplicated_record(x)
    )
    print(flight_df.head())
    tmp = []
    for item in flight_df['duplicated'].tolist():
        tmp += item

    # buckets = lsh_clustering.get_lsh_server().get_counts()
    # print(len(buckets))

    flight_df['buckets'] = flight_df['duplicated'].apply(
        lambda x: '_'.join(x)
    )
    print(flight_df.head())
    unique_labels = flight_df['buckets'].unique().tolist()
    print("number buckets %s" % len(unique_labels))
    print(flight_df.groupby('buckets').size())
    n_curve_per_bucket = flight_df.groupby('buckets').size().to_dict()
    def convert_to_cluster_number(bucket_label, n_curve_per_bucket):
        if n_curve_per_bucket[bucket_label] < 10 :
            return -1
        # if len(bucket_label.split("_")) < 10:
        #     return -1
        return unique_labels.index(bucket_label)
    labels = [convert_to_cluster_number(bucket_label, n_curve_per_bucket)
              for bucket_label in flight_df['buckets'].tolist()]


    # exit()
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels)+1)]

    plt.figure(figsize=(20, 10))
    fig = plt.figure(frameon=False)
    fig.set_size_inches(20, 20)
    ax = fig.add_subplot(1, 1, 1)
    # And a corresponding grid
    ax.grid(which='both')
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    for index, code in enumerate(flight_df['idx'].tolist()):
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
    # grid group reduce size
    # lats, lons = zip(*centermost_points)
    # ax.scatter(
    #     x=lons,  # x axis
    #     y=lats,  # y axis
    #     alpha=0.9,
    #     label=-1,
    #     color='black',
    #     cmap=plt.cm.jet
    # )
    # export images
    silhouette = None
    plt.savefig(
        "../tmp/{}_{}_lsh_sil_{}.png".format(
            "tracks_2015_06_01-001", airport_code, silhouette
        )
    )

    exit(0)

    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({lat_label: lats, lon_label: lons})
    rs = rep_points.apply(
        lambda row: flight_df[(flight_df[lat_label] == row[lat_label]) &
                       (flight_df[lon_label] == row[lon_label])].iloc[0],
        axis=1
    )

    fig, ax = plt.subplots(figsize=[10, 6])
    rs_scatter = ax.scatter(
        x=rs[lon_label], y=rs[lat_label],
        c='#99cc99', edgecolor='None', alpha=0.7, s=120
    )
    df_scatter = ax.scatter(
        x=df[lon_label], y=df[lat_label],
        c='k', alpha=0.9, s=3
    )
    ax.set_title('Full data set vs DBSCAN reduced set')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(
        [df_scatter, rs_scatter],
        ['Full set', 'Reduced set'],
        loc='upper right'
    )
    plt.savefig(
        "../tmp/{}_{}_lsh_reduce.png".format(
            "tracks_2015_06_01-001", airport_code
        )
    )


if __name__ == '__main__':
    main("/Users/tanthm/jvn_data/NTU/tracks_2015_06_01-001.csv")
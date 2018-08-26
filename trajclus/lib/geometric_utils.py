import numpy as np
import traj_dist.distance as tdist
from simplification.cutil import simplify_coords, simplify_coords_vw


KM_PER_RADIAN = 6371.0088


def simplify_coordinator(coord_curve, epsilon=0.0001):
    """

    Args:
        coord_curve (list[list[float, float]]): a list of lat, lon coordinates
        epsilon (float):
    Returns:
        list[list[float, float]]
    """
    coord_curve = np.asarray(coord_curve, order='C')
    return simplify_coords(coord_curve, epsilon)


def build_coordinator_dict(df, label_encoder, flight_ids, max_flights=1000,
                           is_simplify=True):
    """

    Args:
        df:
        label_encoder:
        flight_ids:
        max_flights:

    Returns:

    """
    flight_idx = []
    flight_dicts = {}
    coord_list = []
    count = 1
    for fid in flight_ids:
        if count > max_flights:
            break
        count += 1
        df_min = df[df['Flight_ID'] == fid]
        df_min = df_min.sort_values(by='DRemains', ascending=False)
        encode_id = label_encoder.transform([fid])[0]
        flight_idx.append(encode_id)
        coords = df_min.as_matrix(columns=['Latitude', 'Longitude'])
        if is_simplify:
            coords = simplify_coordinator(coords)
        coord_list.append(coords)
        flight_dicts[encode_id] = coords

    return flight_idx, coord_list, flight_dicts


def compute_distance_between_curves(u, v, algo='directed_hausdorff'):
    """
    Compute distance of 2 curve u, v
    Args:
        u (list[(float,float)]): list of Lat, Lon first curve
        v (list[(float,float)]): list of Lat, Lon second curve
        algo (str): name of algorithm

    Returns:

    """
    if algo == 'frechet':
        return tdist.frechet(u, v)
    else:
        """ compute symmetric Hausdorff distances of curves """
        # D = scipy.spatial.distance.cdist(u, v, 'euclidean')
        # None symmetric Hausdorff distances
        # H1 = np.max(np.min(D, axis=1))
        # H2 = np.max(np.min(D, axis=0))
        # return (H1 + H2) / 2.
        # Find the general (symmetric) Hausdorff distance between two 2-D arrays of coordinates:
        # return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        return tdist.hausdorff(u, v)


def build_matrix_distances(coords=[], dist_type='directed_hausdorff'):
    """
    Construct the matrix distance between every curves (pair-wise distance)

    Args:
        coords (list[list[(float, float)]]): list of Lat, Lon of curves
        dist_type (str): the type of distance need to compute

    Returns:
        (numpy-array): matrix distance

    """
    if dist_type not in ['directed_hausdorff', 'frechet']:
        return False
    n_curve = len(coords)
    # compute distance matrix
    dist_matrix = np.zeros(shape=(n_curve, n_curve))
    for i in range(0, n_curve - 1):
        for j in range(i + 1, n_curve):
            tmp = compute_distance_between_curves(coords[i], coords[j], dist_type)
            dist_matrix[i, j] = tmp
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
def thresholding_algo(y, lag=30, threshold=5, influence=0):
    """
    Detect peak event

    Args:
        y:
        lag:
        threshold:
        influence:

    Returns:

    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(
        signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

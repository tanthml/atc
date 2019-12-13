import pandas as pd
from sklearn import preprocessing

from trajclus.lib.geometric_utils import simplify_coordinator


def flight_id_encoder(unique_id):
    """
    Encoding flight id to integer number
    Args:
        unique_id (list[str]): list flight id

    Returns:
        le (LabelEncoder):

    """
    le = preprocessing.LabelEncoder()
    le.fit(unique_id)
    return le


def filter_by_airport(df, airport_code, min_dr=0.01, max_dr=2.0):
    """
    Filter data-frame by airport-code and radius
    Args:
        df (pd.DataFrame):
        airport_code (str): Destination code name
        min_dr (float): min value of DRemains
        max_dr (float): max value of DRemains

    Returns:
        flights_to_airport (pd.DataFrame):
    """
    one_airport = df[(df['Destination'] == airport_code)]
    # get fixed
    flights_to_airport = one_airport[
        (min_dr < one_airport['DRemains']) & (one_airport['DRemains'] < max_dr)
    ]

    return flights_to_airport


def build_flight_trajectory_df(flights_to_airport, label_encoder, flight_ids,
                               max_flights=1000, epsilon=None):
    """
    build data-frame contains flight-ID and coordinators of flight trajectories
    Args:
        flights_to_airport (pd.DataFrame):
        label_encoder (LabelEncoder):
        flight_ids (list[str]):
        max_flights (int):
        is_simplify (bool):

    Returns:
        flight_df (pd.DataFrame)
    """
    encoded_idx = []
    trajectories = []
    flight_dicts = {}

    for fid in flight_ids[:max_flights]:
        df_min = flights_to_airport[flights_to_airport['Flight_ID'] == fid]
        df_min = df_min.sort_values(by='DRemains', ascending=False)
        encode_id = label_encoder.transform([fid])[0]
        encoded_idx.append(encode_id)
        coords = df_min[['Latitude', 'Longitude']].values
        flight_dicts[encode_id] = coords
        if epsilon:
            coords = simplify_coordinator(coords, epsilon=epsilon)
        trajectories.append(coords)


    flight_df = pd.DataFrame()
    flight_df['idx'] = encoded_idx
    flight_df['flight_id'] = flight_ids[:max_flights]
    flight_df['trajectory'] = trajectories
    print("Total extracted flights %s" % len(flight_df))

    return flight_df, flight_dicts

import json
from copy import deepcopy

import click
import csv
import pandas as pd

from lib.utils import gen_log_file
logger = gen_log_file(path_to_file='tmp/convert_flight_format.log')


def get_num_lines_in_file(file_path):
    """
    Get number of lines in file

    Args:
        file_path (str): file path

    Returns:
        (int): number of lines

    """
    from subprocess import check_output
    return int(check_output(
        ['wc', '-l', file_path]).split(b' ')[0])


@click.command()
@click.option(
    '--input_path',
    type=str,
    required=True,
    help='Full path to the trajectory file in json format')
def main(input_path):
    logger.info("Filepath : {}".format(input_path))
    flights = []
    flight_keys = [
        "Flight ID",
        "Ident",
        "Origin",
        "Destination",
        "Actual Arrival Time (UTC)"]
    tract_keys = [
        "DRemains",
        "TRemains",
        "TTravelled",
        "Time (UTC)",
        "Latitude",
        "Longitude",
        "Altitude (ft)",
        "Rate",
        "Course",
        "Direction",
        # "Facility Name",
        # "Facility Description",
        # "Estimated Pos.",
    ]
    col_order = [
        'Flight_ID',
        'Ident',
        'Origin',
        'Destination',
        'Actual_Arrival_Time_(UTC)',
        'DRemains',
        'TRemains',
        'TTravelled',
        'Time_(UTC)',
        'Latitude',
        'Longitude',
        "Altitude_(ft)",
        "Rate",
        "Course",
        "Direction",
        # "Facility_Name",
        # "Facility_Description",
        # "Estimated_Pos.",
    ]
    fin = open(input_path.replace('.json', '.csv'), 'w')
    writer = csv.DictWriter(fin, fieldnames=col_order)
    writer.writeheader()
    num_lines = get_num_lines_in_file(input_path)
    logger.info("Total {} records ".format(num_lines))
    flights_id = []
    with open(input_path) as fin:
        for i, line in enumerate(fin):
            # print progress bar
            one_flight = (json.loads(line))
            flight_header = {}
            if one_flight['flight']['Flight ID'] in flights_id:
                print("FlightID overlap: %s" % one_flight['flight']['Flight ID'])
            flights_id.append(one_flight['flight']['Flight ID'])
            for key in flight_keys:
                flight_header[key.replace(' ', '_')] = one_flight['flight'][key]
            for tract in one_flight['track']:
                flight_tract = deepcopy(flight_header)
                for track_key in tract_keys:
                    flight_tract[track_key.replace(' ', '_')] = tract[track_key]
                writer.writerow(flight_tract)
    fin.close()


# @click.command()
# @click.option(
#     '--input_path',
#     type=str,
#     required=True,
#     help='Full path to the trajectory file in json format')
# def main(input_path):
#     logger.info("Processing {}".format(input_path))
#     flights = []
#     flight_append = flights.append
#     with open(input_path) as fin:
#         for line in fin:
#             flight_keys = [
#                 "Flight ID", "Ident", "Origin", "Destination",
#                 "Actual Arrival Time (UTC)"]
#             tract_keys = [
#                 "Time (UTC)",
#                 "Latitude",
#                 "Longitude",
#                 "Altitude (ft)",
#                 "Rate",
#                 # "Course",
#                 # "Direction",
#                 # "Facility Name",
#                 # "Facility Description",
#                 # "Estimated Pos.",
#                 "TTravelled",
#                 "TRemains",
#                 "DRemains"
#             ]
#             one_flight = (json.loads(line))
#             flight_header = {}
#             for key in flight_keys:
#                 flight_header[key.replace(' ', '_')] = one_flight['flight'][key]
#             for tract in one_flight['track']:
#                 flight_tract = deepcopy(flight_header)
#                 for track_key in tract_keys:
#                     flight_tract[track_key.replace(' ', '_')] = tract[track_key]
#                 flight_append(flight_tract)
#
#     ''' Transform to dataframe '''
#     pd.DataFrame(flights).to_csv(
#         input_path.replace('.json', '.csv'),
#         index=False)
#     logger.info("Len: {}".format(len(flights)))
#     logger.info(flights[0])


if __name__ == '__main__':
    main()

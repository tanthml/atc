import json
from copy import deepcopy

import click
import pandas as pd

from utils import gen_log_file
logger = gen_log_file(path_to_file='tmp/convert_flight_format.log')


@click.command()
@click.option(
    '--input_path',
    type=str,
    required=True,
    help='Full path to the trajectory file in json format')
def main(input_path):
    logger.info("Processing {}".format(input_path))
    flights = []
    flight_append = flights.append
    with open(input_path) as fin:
        for line in fin:
            flight_keys = [
                "Flight ID", "Ident", "Origin", "Destination",
                "Actual Arrival Time (UTC)"]
            tract_keys = [
                "Time (UTC)",
                "Latitude",
                "Longitude",
                "Altitude (ft)",
                "Rate",
                # "Course",
                # "Direction",
                # "Facility Name",
                # "Facility Description",
                # "Estimated Pos.",
                "TTravelled",
                "TRemains",
                "DRemains"
            ]
            one_flight = (json.loads(line))
            flight_header = {}
            for key in flight_keys:
                flight_header[key.replace(' ', '_')] = one_flight['flight'][key]
            for tract in one_flight['track']:
                flight_tract = deepcopy(flight_header)
                for track_key in tract_keys:
                    flight_tract[track_key.replace(' ', '_')] = tract[track_key]
                flight_append(flight_tract)

    ''' Transform to dataframe '''
    pd.DataFrame(flights).to_csv(
        input_path.replace('.json', '.csv'),
        index=False)
    logger.info("Len: {}".format(len(flights)))
    logger.info(flights[0])


if __name__ == '__main__':
    main()

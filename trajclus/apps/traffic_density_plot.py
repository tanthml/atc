from time import gmtime, strftime

import click
import pandas as pd

from trajclus.lib.common_utils import gen_log_file
from trajclus.lib.preprocessing_lib import filter_by_airport
from trajclus.lib.plot_utils import traffic_density_plot


logger = gen_log_file(path_to_file='../tmp/traffic_density.log')


def main(input_path, airport_code, date):
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
        min_dr=0.1,
        max_dr=5.0
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
    logger.info("Total # flight ID {} {}".format(len(flight_ids), airport_code))
    print("Total # flight ID {} {}".format(len(flight_ids), airport_code))


@click.command()
@click.option(
    '--input_path',
    type=str,
    required=True,
    help='Full path to the trajectory file in CSV format')
@click.option(
    '--airport_code',
    type=str,
    default='WSSS,VTBS,VVTS',
    help='Air Port Codename')
@click.option(
    '--date',
    type=str,
    default='2016-09-01',
    help='Arrival date')
def main_cli(input_path, airport_code, date):
    airports = airport_code.split(",")
    for airport in airports:
        main(input_path=input_path, airport_code=airport, date=date)


if __name__ == '__main__':
    main_cli()

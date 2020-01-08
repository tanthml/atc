import click
import pandas as pd

from trajclus.lib.common_utils import gen_log_file
from trajclus.lib.preprocessing_lib import filter_by_airport


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
        min_dr=0.0,
        max_dr=5.0,
        filter_date='',
):
    # load raw-data from csv
    logger = gen_log_file(path_to_file='../tmp/extract_data_to_airport{}.log'.format(filter_date))
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
        min_dr=min_dr,
        max_dr=max_dr
    )

    flights_to_airport.to_csv(
        "{}_{}.csv".format(file_name, airport_code), index=False
    )

    logger.info("Encoding flight ID ... %s" % airport_code)
    flight_ids = flights_to_airport['Flight_ID'].unique().tolist()
    logger.info("Total # flight ID {}".format(len(flight_ids)))


@click.command()
@click.option(
    '--input_path',
    type=str,
    required=True,
    help='Full path to the trajectory file in CSV format')
@click.option(
    '--airport_code',
    type=str,
    default='WSSS,VTBS,WMKK',
    # default='WSSS',
    help='Air Port Codename')
@click.option(
    '--dr_range',
    type=str,
    default='0.0,5.0',
    help='distance remains in radius')
@click.option(
    '--filter_date',
    type=str,
    default='',
    help='Filter by date example 2016-09-29')
def main_cli(input_path, airport_code, dr_range, filter_date):
    airports = airport_code.split(",")
    dr_ranges = [float(i) for i in dr_range.split(",")]
    for airport in airports:
        main(
            input_path=input_path,
            airport_code=airport,
            min_dr=dr_ranges[0],
            max_dr=dr_ranges[1],
            filter_date=filter_date,
        )


if __name__ == '__main__':
    main_cli()

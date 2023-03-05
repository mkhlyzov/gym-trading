import pathlib

import pandas

ENDPOINTS = {
    "BITCOIN_USD_1H": "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv"
}


def load_asset_from_disc(asset_name: str) -> pandas.DataFrame:
    """
    > Loads a CSV file from the `data` directory and returns a `pandas.DataFrame` object

    :param asset_name: The name of the asset you want to load
    :type asset_name: str
    :return: A dataframe
    """
    asset_path = (
        pathlib.Path(__file__).resolve().parent / "data" / "".join([asset_name, ".csv"])
    )
    asset_df = pandas.read_csv(asset_path)
    asset_df['date'] = pandas.to_datetime(asset_df['date'])
    asset_df.set_index("date", inplace=True)
    return asset_df


def load_asset_from_internet(asset_name: str) -> pandas.DataFrame:
    """
    > This function takes in a string, and returns a pandas dataframe

    :param asset_name: The name of the asset you want to load
    :type asset_name: str
    :return: A dataframe with the asset data.
    """
    # https://stackoverflow.com/questions/62278538/pd-read-csv-produces-httperror-http-error-403-forbidden
    url = ENDPOINTS.get(asset_name, None)
    storage_options = {"User-Agent": "Mozilla/5.0"}
    asset_df = pandas.read_csv(url, header=1, storage_options=storage_options)
    asset_df['date'] = pandas.to_datetime(asset_df['date'])
    asset_df.sort_values(by=["date"], inplace=True, ignore_index=True)
    asset_df.set_index("date", inplace=True)
    return asset_df


def fetch_asset(asset_name: str) -> None:
    """
    > It downloads a CSV file from the internet, and saves it to a local file

    :param asset_name: The name of the asset you want to fetch
    :type asset_name: str
    """
    asset_df = load_asset_from_internet(asset_name=asset_name)
    to_path = (
        pathlib.Path(__file__).resolve().parent / "data" / "".join([asset_name, ".csv"])
    )
    asset_df.to_csv(to_path)

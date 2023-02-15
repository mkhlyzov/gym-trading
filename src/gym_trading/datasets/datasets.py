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
    asset_path = pathlib.Path(__file__).resolve().parent / "".join([asset_name, ".csv"])
    asset_df = pandas.read_csv(asset_path)
    return asset_df


def load_asset_from_internet(asset_name: str) -> pandas.DataFrame:
    """
    > This function takes in a string, and returns a pandas dataframe
    
    :param asset_name: The name of the asset you want to load
    :type asset_name: str
    :return: A dataframe with the asset data.
    """
    path = ENDPOINTS.get(asset_name, None)
    asset_df = pandas.read_csv(path, header=1)
    asset_df.sort_values(by=["date"], inplace=True, ignore_index=True)
    return asset_df


def fetch_asset(asset_name: str) -> None:
    """
    > It downloads a CSV file from the internet, and saves it to a local file
    
    :param asset_name: The name of the asset you want to fetch
    :type asset_name: str
    """
    asset_df = load_asset_from_internet(asset_name=asset_name)
    to_path = pathlib.Path(__file__).resolve().parent / "".join([asset_name, ".csv"])
    asset_df.to_csv(to_path, index=False)

from .datasets import (fetch_asset, load_asset_from_disc,
                       load_asset_from_internet)

try:
    BITCOIN_USD_1H = load_asset_from_disc("BITCOIN_USD_1H")
except FileNotFoundError:
    fetch_asset("BITCOIN_USD_1H")
    BITCOIN_USD_1H = load_asset_from_disc("BITCOIN_USD_1H")

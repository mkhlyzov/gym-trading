import pandas as pd

try:
    BITCOIN_USD_1H = pd.read_csv(
        'https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_1h.csv',
        header=1
    )
    BITCOIN_USD_1H.sort_values(by=['date'], inplace=True, ignore_index=True)
except Exception:
    BITCOIN_USD_1H = pd.DataFrame()

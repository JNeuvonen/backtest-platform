import json


class RowScrapeJob:
    def __init__(
        self,
        id,
        url,
        margin_type,
        prefix,
        columns_to_drop,
        dataseries_interval,
        klines,
        pair,
        candle_interval,
        market_type,
        path,
        ongoing,
        finished,
        tries,
    ):
        self.id = id
        self.url = url
        self.margin_type = margin_type
        self.prefix = prefix
        self.columns_to_drop = json.loads(
            columns_to_drop
        )  # Assuming this is a JSON string
        self.dataseries_interval = dataseries_interval
        self.klines = klines
        self.pair = pair
        self.candle_interval = candle_interval
        self.market_type = market_type
        self.path = path
        self.ongoing = ongoing
        self.finished = finished
        self.tries = tries

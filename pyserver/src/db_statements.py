DELETE_SCRAPE_JOB = "DELETE FROM binance_scrape_job WHERE id = ?"
INSERT_SCRAPE_JOB = """
                        INSERT INTO binance_scrape_job (
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
                                    tries
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
CREATE_SCRAPE_JOB_TABLE = """
                        CREATE TABLE IF NOT EXISTS binance_scrape_job (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            url TEXT,
                            margin_type TEXT,
                            prefix TEXT,
                            columns_to_drop TEXT,
                            dataseries_interval TEXT,
                            klines TEXT,
                            pair TEXT,
                            candle_interval TEXT,
                            market_type TEXT,
                            path TEXT,
                            ongoing INTEGER,
                            finished INTEGER,
                            tries INTEGER
                        )
                            """
UPDATE_SCRAPE_JOB_TRIES = "UPDATE binance_scrape_job SET tries = ? WHERE id = ?"
UPDATE_SCRAPE_JOB_ONGOING = "UPDATE binance_scrape_job SET ongoing = ? where id = ?"

import { CONSTANTS } from "../utils/constants";

const API = {
  tables: "/tables",
  binance: {
    get_all_tickers: "/binance/get-all-tickers",
  },
};

const BASE_URL = CONSTANTS.base_url;

export const URLS = {
  get_tables: BASE_URL + API.tables,
  binance_get_all_tickers: BASE_URL + API.binance.get_all_tickers,
};

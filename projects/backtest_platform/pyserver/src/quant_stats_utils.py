import pandas as pd
from bs4 import BeautifulSoup
from constants import BACKTEST_REPORT_HTML_PATH
import quantstats_lumi as qs
from request_types import BodyCreateManualBacktest

from utils import binary_search_on_pd_timeseries, read_file_to_string


PRE_STYLE_BASE = "background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600; word-wrap: break-word; overflow-wrap: break-word; white-space: pre-wrap; width: 100%"
PRE_STYLE_BLUE = f"{PRE_STYLE_BASE} border-left: 3px solid #36a2eb;"
PRE_STYLE_RED = f"{PRE_STYLE_BASE} border-left: 3px solid #eb4036;"


def generate_quant_stats_report_html(balance_history, backtestInfo, periods_per_year):
    balance_history_df = pd.DataFrame(balance_history)
    balance_history_df["kline_open_time"] = pd.to_datetime(
        balance_history_df["kline_open_time"], unit="ms"
    )

    balance_history_df.set_index("kline_open_time", inplace=True)

    strat_returns = get_df_returns(balance_history_df, "portfolio_worth")

    qs.reports.html(
        strat_returns,
        output=BACKTEST_REPORT_HTML_PATH,
        title="Backtest Performance Report",
        periods_per_year=periods_per_year,
        cagr_periods_per_year=365,
    )
    update_backtest_report_html(backtestInfo)


def get_df_returns(df, ret_col):
    return qs.utils.to_returns(df[ret_col]).dropna()


def append_custom_div(soup, style, content):
    pre_tag = soup.new_tag("pre", style=style)
    pre_tag.string = content
    return pre_tag


def enhance_report_with_details(soup, backtest_info, symbols=[]):
    container = soup.find("div", class_="container")
    trading_criteria_div = soup.new_tag("div")
    header_tag = soup.new_tag("h1")

    if len(symbols) == 0:
        header_tag.string = f"{backtest_info.name} strategy"
    else:
        header_tag.string = f"Trading rules are applied to {len(symbols)} pairs"
    trading_criteria_div.append(header_tag)

    if len(symbols) > 0:
        trading_criteria_div.append(
            append_custom_div(soup, PRE_STYLE_BASE, f"{', '.join(symbols)}")
        )

    strategy_type = "Short" if backtest_info.is_short_selling_strategy else "Long"

    trading_criteria_div.append(
        append_custom_div(soup, PRE_STYLE_BASE, f"{strategy_type} strategy")
    )

    trading_criteria_div.append(
        append_custom_div(soup, PRE_STYLE_BLUE, backtest_info.open_trade_cond)
    )
    trading_criteria_div.append(
        append_custom_div(soup, PRE_STYLE_RED, backtest_info.close_trade_cond)
    )

    if backtest_info.use_time_based_close:
        content = f"Using time based close, position is holded for maximum of {backtest_info.klines_until_close} candles"
        trading_criteria_div.append(append_custom_div(soup, PRE_STYLE_BASE, content))
    if backtest_info.use_stop_loss_based_close:
        content = (
            f"Using stop loss based close, {backtest_info.stop_loss_threshold_perc}%"
        )
        trading_criteria_div.append(append_custom_div(soup, PRE_STYLE_BASE, content))
    if backtest_info.use_profit_based_close:
        content = (
            f"Using profit based close, {backtest_info.take_profit_threshold_perc}%"
        )
        trading_criteria_div.append(append_custom_div(soup, PRE_STYLE_BASE, content))

    assumptions_header = soup.new_tag("h1", style="margin-top: 16px;")
    assumptions_header.string = "Backtest assumptions"
    trading_criteria_div.append(assumptions_header)

    trading_criteria_div.append(
        append_custom_div(
            soup, PRE_STYLE_BASE, f"Trading fees: {backtest_info.trading_fees_perc}%"
        )
    )
    trading_criteria_div.append(
        append_custom_div(
            soup, PRE_STYLE_BASE, f"Slippage: {backtest_info.slippage_perc}%"
        )
    )

    if backtest_info.is_short_selling_strategy:
        content = f"Short fee hourly: {backtest_info.short_fee_hourly}%"
        trading_criteria_div.append(append_custom_div(soup, PRE_STYLE_BASE, content))

    container.insert(0, trading_criteria_div)


def update_backtest_report_html(backtest_info, symbols=[]):
    report_html_str = read_file_to_string(BACKTEST_REPORT_HTML_PATH)

    soup = BeautifulSoup(report_html_str, "html.parser")

    enhance_report_with_details(soup, backtest_info, symbols)

    with open(BACKTEST_REPORT_HTML_PATH, "w", encoding="utf-8") as file:
        file.write(str(soup))


def generate_combined_report(
    dict_of_returns,
    id_to_dataset_name_map,
    periods_per_year,
    backtestInfo,
):
    longest_time_series = max(dict_of_returns.items(), key=lambda x: len(x[1]))[1]
    longest_time_series_key = max(
        dict_of_returns, key=lambda k: len(dict_of_returns[k])
    )
    num_strategies = len(dict_of_returns)

    results_df = pd.DataFrame(columns=["kline_open_time", "portfolio_worth"])
    current_equity = 10000

    data = []

    for index, value in longest_time_series.items():
        round_helper = {
            key: current_equity / num_strategies for key in dict_of_returns.keys()
        }

        round_helper[longest_time_series_key] *= value

        for key, series_of_rets in dict_of_returns.items():
            if key == longest_time_series_key:
                continue

            other_value = binary_search_on_pd_timeseries(series_of_rets, index)
            if other_value is not None:
                round_helper[key] *= other_value

        current_equity = sum(round_helper.values())

        data.append({"kline_open_time": index, "portfolio_worth": current_equity})

    results_df = pd.DataFrame(data)

    results_df.set_index("kline_open_time", inplace=True)

    returns = get_df_returns(results_df, "portfolio_worth")
    symbols = []

    for key, value in id_to_dataset_name_map.items():
        symbols.append(value)

    qs.reports.html(
        returns,
        output=BACKTEST_REPORT_HTML_PATH,
        title="Backtest Performance Report",
        periods_per_year=periods_per_year,
        cagr_periods_per_year=365,
    )

    update_backtest_report_html(
        BodyCreateManualBacktest(
            backtest_data_range=[
                backtestInfo["backtest_range_start"],
                backtestInfo["backtest_range_end"],
            ],
            open_trade_cond=backtestInfo["open_trade_cond"],
            close_trade_cond=backtestInfo["close_trade_cond"],
            is_short_selling_strategy=backtestInfo["is_short_selling_strategy"],
            use_stop_loss_based_close=backtestInfo["use_stop_loss_based_close"],
            use_time_based_close=backtestInfo["use_time_based_close"],
            use_profit_based_close=backtestInfo["use_profit_based_close"],
            dataset_id=backtestInfo["dataset_id"],
            trading_fees_perc=backtestInfo["trading_fees_perc"],
            slippage_perc=backtestInfo["slippage_perc"],
            short_fee_hourly=backtestInfo["short_fee_hourly"],
            take_profit_threshold_perc=backtestInfo["take_profit_threshold_perc"],
            stop_loss_threshold_perc=backtestInfo["stop_loss_threshold_perc"],
            name=backtestInfo["name"],
            klines_until_close=backtestInfo["klines_until_close"],
        ),
        symbols,
    )

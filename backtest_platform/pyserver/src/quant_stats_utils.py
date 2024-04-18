import pandas as pd
from bs4 import BeautifulSoup
import quantstats as qs

from utils import read_file_to_string


BACKTEST_REPORT_HTML_PATH = "backtest_report.html"


def generate_quant_stats_report_html(balance_history, backtestInfo):
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
    )

    report_html_str = read_file_to_string(BACKTEST_REPORT_HTML_PATH)
    soup = BeautifulSoup(report_html_str, "html.parser")
    container = soup.find("div", class_="container")

    trading_criteria_div = soup.new_tag("div")
    header_tag = soup.new_tag("h1")

    pre_strategy_type = soup.new_tag(
        "pre",
        style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
    )

    pre_tag_open_trade_cond = soup.new_tag(
        "pre",
        style="background-color: #f9f9f9; border-left: 3px solid #36a2eb; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
    )
    pre_tag_close_trade_cond = soup.new_tag(
        "pre",
        style="background-color: #f9f9f9; border-left: 3px solid #eb4036; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
    )

    header_tag.string = "Strategy details"
    pre_strategy_type.string = (
        "Short" if backtestInfo.is_short_selling_strategy else "Long" + " strategy"
    )
    pre_tag_open_trade_cond.string = backtestInfo.open_trade_cond
    pre_tag_close_trade_cond.string = backtestInfo.close_trade_cond

    trading_criteria_div.append(header_tag)
    trading_criteria_div.append(pre_strategy_type)
    trading_criteria_div.append(pre_tag_open_trade_cond)
    trading_criteria_div.append(pre_tag_close_trade_cond)

    if backtestInfo.use_time_based_close:
        pre_tag = soup.new_tag(
            "pre",
            style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
        )
        pre_tag.string = (
            "Using time based close, position is holded for maximum of "
            + str(backtestInfo.klines_until_close)
            + " candles"
        )
        trading_criteria_div.append(pre_tag)

    if backtestInfo.use_stop_loss_based_close:
        pre_tag = soup.new_tag(
            "pre",
            style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
        )
        pre_tag.string = (
            "Using stop loss based close, "
            + str(backtestInfo.stop_loss_threshold_perc)
            + "%"
        )
        trading_criteria_div.append(pre_tag)

    if backtestInfo.use_profit_based_close:
        pre_tag = soup.new_tag(
            "pre",
            style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
        )
        pre_tag.string = (
            "Using profit based close, "
            + str(backtestInfo.take_profit_threshold_perc)
            + "%"
        )
        trading_criteria_div.append(pre_tag)

    backtest_assumptions_header = soup.new_tag("h1", style="margin-top: 16px;")
    backtest_assumptions_header.string = "Backtest assumptions"
    trading_criteria_div.append(backtest_assumptions_header)

    pre_tag_trading_fees = soup.new_tag(
        "pre",
        style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
    )

    pre_tag_trading_fees.string = (
        "Trading fees: " + str(backtestInfo.trading_fees_perc) + "%"
    )

    pre_tag_slippage = soup.new_tag(
        "pre",
        style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
    )

    pre_tag_slippage.string = "Slippage: " + str(backtestInfo.slippage_perc) + "%"

    trading_criteria_div.append(pre_tag_trading_fees)
    trading_criteria_div.append(pre_tag_slippage)

    if backtestInfo.is_short_selling_strategy:
        pre_tag = soup.new_tag(
            "pre",
            style="background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;",
        )

        pre_tag.string = "Short fee hourly: " + str(backtestInfo.short_fee_hourly) + "%"
        trading_criteria_div.append(pre_tag)

    container.insert(0, trading_criteria_div)

    with open(BACKTEST_REPORT_HTML_PATH, "w", encoding="utf-8") as file:
        file.write(str(soup))


def get_df_returns(df, ret_col):
    return qs.utils.to_returns(df[ret_col]).dropna()


def gen_report():
    pass

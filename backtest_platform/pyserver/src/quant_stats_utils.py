import pandas as pd
from bs4 import BeautifulSoup
import quantstats as qs

from utils import read_file_to_string


BACKTEST_REPORT_HTML_PATH = "backtest_report.html"


PRE_STYLE_BASE = "background-color: #f9f9f9; padding: 10px; margin: 10px; font-family: Consolas, 'Courier New', Courier, monospace; color: #333; font-weight: 600;"
PRE_STYLE_BLUE = f"{PRE_STYLE_BASE} border-left: 3px solid #36a2eb;"
PRE_STYLE_RED = f"{PRE_STYLE_BASE} border-left: 3px solid #eb4036;"


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
    update_backtest_report_html(backtestInfo)


def get_df_returns(df, ret_col):
    return qs.utils.to_returns(df[ret_col]).dropna()


def append_custom_div(soup, style, content):
    pre_tag = soup.new_tag("pre", style=style)
    pre_tag.string = content
    return pre_tag


def enhance_report_with_details(soup, backtest_info):
    container = soup.find("div", class_="container")
    trading_criteria_div = soup.new_tag("div")
    header_tag = soup.new_tag("h1")
    header_tag.string = "Strategy details"
    trading_criteria_div.append(header_tag)

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


def update_backtest_report_html(backtest_info):
    report_html_str = read_file_to_string(BACKTEST_REPORT_HTML_PATH)
    soup = BeautifulSoup(report_html_str, "html.parser")

    enhance_report_with_details(soup, backtest_info)

    with open(BACKTEST_REPORT_HTML_PATH, "w", encoding="utf-8") as file:
        file.write(str(soup))

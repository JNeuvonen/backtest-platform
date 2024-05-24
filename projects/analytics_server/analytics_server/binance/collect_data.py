import time
from common_python.binance.funcs import get_binance_acc_balance_snapshot
from common_python.pred_serv_models.strategy import StrategyQuery
from common_python.pred_serv_models.longshortpair import LongShortPairQuery
from common_python.pred_serv_models.balance_snapshot import BalanceSnapshotQuery
from common_python.log import LogExceptionContext


def collect_data_loop(stop_event):
    while not stop_event.is_set():
        with LogExceptionContext(re_raise=False):
            balance_snapshot = get_binance_acc_balance_snapshot()
            strategies_in_pos = StrategyQuery.get_strategies_in_position()
            num_ls_strategies_in_pos = LongShortPairQuery.count_pairs_in_position()

            num_long_strats = 0
            num_short_strats = 0

            for item in strategies_in_pos:
                if item.is_short_selling_strategy:
                    num_short_strats += 1
                else:
                    num_long_strats += 1

            BalanceSnapshotQuery.create_entry(
                {
                    "value": balance_snapshot["nav_usdt"],
                    "debt": balance_snapshot["liability_usdt"],
                    "long_assets_value": balance_snapshot["total_asset_usdt"],
                    "margin_level": balance_snapshot["margin_level"],
                    "btc_price": balance_snapshot["btc_price"],
                    "num_long_positions": num_long_strats,
                    "num_short_positions": num_short_strats,
                    "num_ls_positions": num_ls_strategies_in_pos,
                }
            )
            time.sleep(15 * 60)

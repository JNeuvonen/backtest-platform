import { Stat, StatLabel } from "@chakra-ui/react";
import {
  findCurrentPrice,
  getNumberDisplayColor,
  isSame24h,
  isSameMonth,
  isSameYear,
  roundNumberFloor,
  safeDivide,
  TRADE_DIRECTIONS,
} from "src/common_js";
import { useAppContext } from "src/context";
import {
  useBinanceSpotPriceInfo,
  useLatestBalanceSnapshot,
  useStrategiesQuery,
} from "src/http/queries";
import { COLOR_BG_PRIMARY, COLOR_CONTENT_PRIMARY } from "src/theme";
import { SIDENAV_DEFAULT_WIDTH } from "src/utils";

export const BottomInfoFooter = () => {
  const { isMobileLayout } = useAppContext();
  const latestSnapshotQuery = useLatestBalanceSnapshot();
  const strategiesQuery = useStrategiesQuery();
  const binancePriceQuery = useBinanceSpotPriceInfo();

  if (
    !strategiesQuery.data ||
    strategiesQuery.isLoading ||
    !binancePriceQuery.data ||
    !latestSnapshotQuery.data
  ) {
    return (
      <div
        style={{
          bottom: 0,
          position: "fixed",
          height: "35px",
          width: "100vw",
          left: isMobileLayout ? 0 : SIDENAV_DEFAULT_WIDTH,
          background: COLOR_BG_PRIMARY,
          display: "flex",
          alignItems: "center",
        }}
      ></div>
    );
  }

  const getInfoDict = () => {
    let totalLongs = 0;
    let totalShorts = 0;
    let unrealizedProfit = 0;
    let mtdTotal = 0;
    let ytdTotal = 0;
    let tradesPast24h = 0;

    const trades = strategiesQuery.data.trades;

    if (!trades) {
      return null;
    }

    trades.forEach((item) => {
      const latestPrice = findCurrentPrice(
        item.symbol,
        binancePriceQuery.data || [],
      );
      const createdAt = new Date(item.created_at);
      const presentDate = new Date();

      let tradeUnrealizedProfit = 0;
      let tradeRealizedProfit = 0;
      let tradeValue = 0;

      if (!latestPrice) return;

      if (!item.close_price) {
        if (item.direction === TRADE_DIRECTIONS.long) {
          tradeValue = latestPrice * item.quantity;
          tradeRealizedProfit = (latestPrice - item.open_price) * item.quantity;

          totalLongs += tradeValue;
        } else {
          tradeValue = latestPrice * item.quantity;
          tradeRealizedProfit = (item.open_price - latestPrice) * item.quantity;
          totalShorts += tradeValue;
        }
      } else {
        if (item.direction === TRADE_DIRECTIONS.long) {
          tradeUnrealizedProfit = item.net_result;
        } else {
          tradeUnrealizedProfit = item.net_result;
        }
      }

      if (isSameMonth(createdAt, presentDate)) {
        mtdTotal += tradeUnrealizedProfit + tradeRealizedProfit;
      }

      if (isSameYear(createdAt, presentDate)) {
        ytdTotal += tradeUnrealizedProfit + tradeRealizedProfit;
      }

      if (isSame24h(createdAt, presentDate)) {
        tradesPast24h += 1;
      }
    });

    return {
      totalLongs,
      totalShorts,
      unrealizedProfit,
      mtdTotal,
      ytdTotal,
      tradesPast24h,
    };
  };

  const infoDict = getInfoDict();

  if (!infoDict) {
    return null;
  }
  return (
    <div
      style={{
        bottom: 0,
        position: "fixed",
        height: "35px",
        width: `calc(100vw - ${SIDENAV_DEFAULT_WIDTH}px - 15px)`,
        left: isMobileLayout ? 0 : SIDENAV_DEFAULT_WIDTH,
        background: COLOR_BG_PRIMARY,
        justifyContent: "space-between",
        display: "flex",
        alignItems: "center",
        paddingLeft: "16px",
        paddingRight: "16px",
      }}
    >
      <div style={{ display: "flex", gap: "8px" }}>
        <div>
          <Stat>
            <StatLabel>
              Acc value: {roundNumberFloor(latestSnapshotQuery.data.value, 2)}$
            </StatLabel>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>
              Longs:{" "}
              {roundNumberFloor(
                safeDivide(
                  infoDict.totalLongs * 100,
                  latestSnapshotQuery.data.value,
                  0,
                ),
                2,
              )}
              %
            </StatLabel>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>
              Shorts:{" "}
              {roundNumberFloor(
                safeDivide(
                  infoDict.totalShorts * 100,
                  latestSnapshotQuery.data.value,
                  0,
                ),
                2,
              )}
              %
            </StatLabel>
          </Stat>
        </div>
        <div>
          <Stat color={COLOR_CONTENT_PRIMARY}>
            <StatLabel>Trades past 24h: {infoDict.tradesPast24h}</StatLabel>
          </Stat>
        </div>
      </div>
      <div style={{ display: "flex", gap: "8px" }}>
        <div>
          <Stat
            color={getNumberDisplayColor(
              infoDict.ytdTotal,
              COLOR_CONTENT_PRIMARY,
            )}
          >
            <StatLabel>
              YTD: {roundNumberFloor(infoDict.ytdTotal, 2)}$
            </StatLabel>
          </Stat>
        </div>
        <div>
          <Stat
            color={getNumberDisplayColor(
              infoDict.mtdTotal,
              COLOR_CONTENT_PRIMARY,
            )}
          >
            <StatLabel>
              MTD: {roundNumberFloor(infoDict.mtdTotal, 2)}$
            </StatLabel>
          </Stat>
        </div>
      </div>
    </div>
  );
};

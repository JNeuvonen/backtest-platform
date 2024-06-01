import {
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
} from "@chakra-ui/react";
import { ColDef, ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import {
  BinanceSymbolPrice,
  findCurrentPrice,
  formatSecondsIntoTime,
  getNumberDisplayColor,
  LongShortGroup,
  LongShortTicker,
  LongShortTrade,
  roundNumberFloor,
  safeDivide,
  StrategiesResponse,
  Trade,
  TRADE_DIRECTIONS,
} from "common_js";
import { Link } from "react-router-dom";
import { ChakraCard } from "src/components/chakra";
import {
  useBinanceSpotPriceInfo,
  useLatestBalanceSnapshot,
} from "src/http/queries";
import { COLOR_CONTENT_PRIMARY } from "src/theme";
import { getLsStrategyPath } from "src/utils";
import { profitColumnCellRenderer } from "./directionaltab";

const strategyNameCellRenderer = (params: ICellRendererParams) => {
  return (
    <Link
      to={getLsStrategyPath(params.value).toLowerCase()}
      className="link-default"
    >
      {params.value}
    </Link>
  );
};

const COLUMN_DEFS: ColDef[] = [
  {
    headerName: "Strategy name",
    field: "name",
    sortable: true,
    editable: false,
    cellRenderer: strategyNameCellRenderer,
  },
  {
    headerName: "Universe size",
    field: "size",
    sortable: true,
    editable: false,
    aggFunc: "sum",
  },
  {
    headerName: "Open trades",
    field: "openTrades",
    sortable: true,
    editable: false,
    aggFunc: "sum",
  },
  {
    headerName: "Completed trades",
    field: "closedTrades",
    sortable: true,
    editable: false,
    aggFunc: "sum",
  },
  {
    headerName: "Realized profit",
    field: "netResult",
    sortable: true,
    editable: false,
    cellRenderer: profitColumnCellRenderer,
    aggFunc: "sum",
  },
  {
    headerName: "Unrealized profit",
    field: "cumulativeUnrealizedProfit",
    sortable: true,
    editable: false,
    cellRenderer: profitColumnCellRenderer,
    aggFunc: "sum",
  },
  {
    headerName: "Mean trade (%)",
    field: "meanTradeResultPerc",
    sortable: true,
    editable: false,
  },
];

const getLsStrategyTableRowFields = (
  strategy: LongShortGroup,
  trades: Trade[],
  longshortTickers: LongShortTicker[],
  binanceSymbolPrices: BinanceSymbolPrice[],
  completedLongShortTrades: LongShortTrade[],
) => {
  let numSymbols = 0;
  let openTrades = 0;
  let completedTrades = 0;
  let cumulativeNetResult = 0;
  let cumulativeNetPercResult = 0;
  let unrealizedProfit = 0;
  let valueInLongPositions = 0;
  let valueInShortPositions = 0;
  let cumulativePosHoldTime = 0;

  completedLongShortTrades.forEach((item) => {
    if (item.long_short_group_id === strategy.id) {
      cumulativeNetResult += item.net_result;
      cumulativeNetPercResult += item.percent_result;
      completedTrades += 1;
      cumulativePosHoldTime += item.close_time_ms - item.open_time_ms;
    }
  });

  longshortTickers.forEach((item) => {
    if (item.long_short_group_id === strategy.id) {
      numSymbols += 1;
    }
  });

  trades.forEach((item) => {
    if (item.pair_trade_group_id === strategy.id) {
      if (!item.percent_result) {
        openTrades += 1;

        const latestPrice = findCurrentPrice(item.symbol, binanceSymbolPrices);

        if (!latestPrice) {
          return;
        }

        if (item.direction === TRADE_DIRECTIONS.long) {
          unrealizedProfit += (latestPrice - item.open_price) * item.quantity;
          valueInLongPositions += latestPrice * item.quantity;
        } else {
          unrealizedProfit += (item.open_price - latestPrice) * item.quantity;
          valueInShortPositions += latestPrice * item.quantity;
        }
      }
    }
  });

  return {
    size: numSymbols,
    openTrades,
    closedTrades: completedTrades,
    netResult: cumulativeNetResult,
    meanTradeResultPerc:
      completedTrades !== 0
        ? roundNumberFloor(cumulativeNetPercResult / completedTrades, 2)
        : 0,
    cumulativeUnrealizedProfit: unrealizedProfit,
    valueInLongPositions,
    valueInShortPositions,
    cumulativePosHoldTime,
  };
};

export const PairTradeTab = ({
  strategiesRes,
}: {
  strategiesRes: StrategiesResponse | undefined;
}) => {
  const binancePriceQuery = useBinanceSpotPriceInfo();
  const latestSnapshotQuery = useLatestBalanceSnapshot();

  if (!strategiesRes || !binancePriceQuery.data || !latestSnapshotQuery.data) {
    return <Spinner />;
  }

  const getRows = () => {
    const ret = [];
    const lsStrategies = strategiesRes.ls_strategies;

    if (lsStrategies === undefined) {
      return [];
    }

    lsStrategies.forEach((item) => {
      const lsStrategy = {
        name: item.name.replace("{SYMBOL}_", ""),
        ...getLsStrategyTableRowFields(
          item,
          strategiesRes.trades,
          strategiesRes.ls_tickers,
          binancePriceQuery.data || [],
          strategiesRes.completed_longshort_trades,
        ),
      };

      ret.push(lsStrategy);
    });

    return ret.sort((a, b) => b.netResult - a.netResult);
  };

  const getInfoDict = () => {
    let realizedProfit = 0;
    let unrealizedProfit = 0;
    let symbols = 0;
    let cumulativeValueInOpenLongs = 0;
    let cumulativeValueInOpenShorts = 0;
    let openTrades = 0;
    let cumulativePosHoldTimeMs = 0;
    let totalCompletedTrades = strategiesRes.completed_longshort_trades.length;
    let cumulativeMeanTradeResPerc = 0;

    strategiesRes.completed_longshort_trades.forEach((item) => {
      realizedProfit += item.net_result;
      cumulativeMeanTradeResPerc += item.percent_result;
    });

    const rows = getRows();

    rows.forEach((item) => {
      unrealizedProfit += item.cumulativeUnrealizedProfit;
      symbols += item.size;
      cumulativeValueInOpenShorts += item.valueInShortPositions;
      cumulativeValueInOpenLongs += item.valueInLongPositions;
      cumulativePosHoldTimeMs += item.cumulativePosHoldTime;
      openTrades += item.openTrades;
    });

    return {
      realizedProfit,
      unrealizedProfit,
      symbols,
      cumulativeValueInOpenLongs,
      cumulativeValueInOpenShorts,
      meanPosTime: formatSecondsIntoTime(
        safeDivide(cumulativePosHoldTimeMs, totalCompletedTrades, 0) / 1000,
      ),
      meanTradeResPerc: safeDivide(
        cumulativeMeanTradeResPerc,
        totalCompletedTrades,
        0,
      ),
      openTrades,
      totalCompletedTrades,
    };
  };

  const infoDict = getInfoDict();

  return (
    <div>
      <Heading size={"lg"}>Pair-trade strategies</Heading>

      <div style={{ marginTop: "16px" }}>
        <ChakraCard heading={<Heading size={"md"}>Breakdown</Heading>}>
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Realized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    infoDict.realizedProfit,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  ${roundNumberFloor(infoDict.realizedProfit, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Unrealized profit</StatLabel>
                <StatNumber
                  color={getNumberDisplayColor(
                    infoDict.unrealizedProfit,
                    COLOR_CONTENT_PRIMARY,
                  )}
                >
                  ${roundNumberFloor(infoDict.unrealizedProfit, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Total symbols</StatLabel>
                <StatNumber>{infoDict.symbols}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Value in longs</StatLabel>
                <StatNumber>
                  {roundNumberFloor(
                    safeDivide(
                      infoDict.cumulativeValueInOpenLongs,
                      latestSnapshotQuery.data.value,
                      0,
                    ) * 100,
                    2,
                  )}
                  %
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Value in shorts</StatLabel>
                <StatNumber>
                  {roundNumberFloor(
                    safeDivide(
                      infoDict.cumulativeValueInOpenShorts,
                      latestSnapshotQuery.data.value,
                      0,
                    ) * 100,
                    2,
                  )}
                  %
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Completed trades</StatLabel>
                <StatNumber>{infoDict.totalCompletedTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Open positions</StatLabel>
                <StatNumber>{infoDict.openTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Mean trade result</StatLabel>
                <StatNumber>
                  {roundNumberFloor(infoDict.meanTradeResPerc, 2)}%
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Mean pos. time</StatLabel>
                <StatNumber>{infoDict.meanPosTime}</StatNumber>
              </Stat>
            </div>
          </div>
        </ChakraCard>
      </div>
      <div
        className="ag-theme-alpine-dark"
        style={{
          width: "100%",
          height: "calc(100vh - 170px)",
          marginTop: "16px",
        }}
      >
        <AgGridReact
          pagination={true}
          columnDefs={COLUMN_DEFS}
          paginationAutoPageSize={true}
          rowData={getRows()}
        />
      </div>
    </div>
  );
};

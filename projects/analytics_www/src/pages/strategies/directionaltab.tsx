import {
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Text,
} from "@chakra-ui/react";
import { ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import {
  BinanceSymbolPrice,
  findCurrentPrice,
  getNumberDisplayColor,
  roundNumberFloor,
  safeDivide,
  StrategiesResponse,
  Strategy,
  StrategyGroup,
  Trade,
  formatSecondsIntoTime,
} from "common_js";
import { Link } from "react-router-dom";
import { ChakraCard } from "src/components/chakra";
import {
  useBinanceSpotPriceInfo,
  useLatestBalanceSnapshot,
} from "src/http/queries";
import { COLOR_CONTENT_PRIMARY } from "src/theme";
import { getStrategyPath } from "src/utils";

const strategyNameCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return;
  }
  return (
    <Link
      to={getStrategyPath(params.value.toLowerCase())}
      className="link-default"
    >
      {params.value}
    </Link>
  );
};

export const profitColumnCellRenderer = (params: ICellRendererParams) => {
  return (
    <Text color={getNumberDisplayColor(params.value, COLOR_CONTENT_PRIMARY)}>
      {roundNumberFloor(params.value, 2)}$
    </Text>
  );
};

export const percColumnCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return null;
  }

  return (
    <Text color={getNumberDisplayColor(params.value, COLOR_CONTENT_PRIMARY)}>
      {roundNumberFloor(params.value, 2)}%
    </Text>
  );
};

const COLUMN_DEFS = [
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
  },
  {
    headerName: "Open trades",
    field: "openTrades",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Completed trades",
    field: "closedTrades",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Realized profit",
    field: "netResult",
    sortable: true,
    editable: false,
    cellRenderer: profitColumnCellRenderer,
  },
  {
    headerName: "Unrealized profit",
    field: "cumulativeUnrealizedProfit",
    sortable: true,
    editable: false,
    cellRenderer: profitColumnCellRenderer,
  },
  {
    headerName: "Mean trade (%)",
    field: "meanTradeResultPerc",
    sortable: true,
    editable: false,
  },
];

const getStrategyGroupFields = (
  strategyGroup: StrategyGroup,
  strategies: Strategy[],
  trades: Trade[],
  binanceSymbolPrices: BinanceSymbolPrice[],
) => {
  let universeSize = 0;
  let openTrades = 0;
  let netResultTrades = 0;
  let numTrades = 0;
  let cumulativePercResult = 0;
  let completedTrades = 0;
  let cumulativeUnrealizedProfit = 0;
  let valueInOpenPositions = 0;
  let cumulativePosHoldTime = 0;

  let isLongStrategy = true;

  const strategyIdToStrategyGroupIdMap: { [key: number]: number } = {};

  strategies.forEach((item) => {
    if (item.strategy_group_id === strategyGroup.id) {
      universeSize += 1;
      if (item.is_short_selling_strategy) {
        isLongStrategy = false;
      }
    }

    if (item.strategy_group_id) {
      strategyIdToStrategyGroupIdMap[item.id] = item.strategy_group_id;
    }
  });

  trades.forEach((item) => {
    if (item.strategy_id) {
      const strategyGroupId = strategyIdToStrategyGroupIdMap[item.strategy_id];

      if (strategyGroupId === strategyGroup.id) {
        numTrades += 1;
        const latestPrice = findCurrentPrice(item.symbol, binanceSymbolPrices);

        if (
          item.close_price &&
          item.net_result &&
          item.percent_result &&
          item.close_time_ms
        ) {
          netResultTrades += item.net_result;
          cumulativePercResult += item.percent_result;
          completedTrades += 1;

          cumulativePosHoldTime += item.close_time_ms - item.open_time_ms;
        }

        if (item.open_price && !item.close_price) {
          openTrades += 1;
        }

        if (latestPrice && !item.close_price) {
          const unrealizedProfit = isLongStrategy
            ? (latestPrice - item.open_price) * item.quantity
            : (item.open_price - latestPrice) * item.quantity;
          cumulativeUnrealizedProfit += unrealizedProfit;
          valueInOpenPositions += latestPrice * item.quantity;
        }
      }
    }
  });

  return {
    size: universeSize,
    numTrades,
    netResult: roundNumberFloor(netResultTrades, 2),
    openTrades,
    meanTradeResultPerc:
      completedTrades !== 0
        ? roundNumberFloor(cumulativePercResult / completedTrades, 2)
        : 0,
    closedTrades: completedTrades,
    cumulativeUnrealizedProfit,
    valueInOpenPositions,
    cumulativePosHoldTime,
    meanPosHoldTimeMs: safeDivide(cumulativePosHoldTime, completedTrades, 0),
    isLongStrategy,
  };
};

export const DirectionalStrategiesTab = ({
  strategiesRes,
}: {
  strategiesRes: StrategiesResponse | undefined;
}) => {
  const binancePriceQuery = useBinanceSpotPriceInfo();
  const latestSnapshotQuery = useLatestBalanceSnapshot();

  if (!strategiesRes || !binancePriceQuery.data || !latestSnapshotQuery.data) {
    return <Spinner />;
  }

  const getDirectionalStrategies = () => {
    const ret = strategiesRes.strategy_groups;

    if (ret === undefined) {
      return [];
    }

    return ret
      .map((item: StrategyGroup) => {
        return {
          name: item.name,
          ...getStrategyGroupFields(
            item,
            strategiesRes.directional_strategies,
            strategiesRes.trades,
            binancePriceQuery.data || [],
          ),
        };
      })
      .sort((a, b) => b.netResult - a.netResult);
  };

  const getInfoDict = () => {
    const directionalStratsArr = getDirectionalStrategies();

    let unrealizedProfit = 0;
    let realizedProfit = 0;
    let symbols = 0;
    let cumulativeValueInOpenPositions = 0;
    let cumulativeMeanTradeResPerc = 0;
    let totalCompletedTrades = 0;
    let openTrades = 0;
    let cumulativePosHoldTimeMs = 0;
    let totalInLongStrategies = 0;
    let totalInShortStrategies = 0;

    directionalStratsArr.forEach((item) => {
      symbols += item.size;
      cumulativeValueInOpenPositions += item.valueInOpenPositions;
      unrealizedProfit += item.cumulativeUnrealizedProfit;
      realizedProfit += item.netResult;
      cumulativeMeanTradeResPerc += item.meanTradeResultPerc;
      totalCompletedTrades += item.closedTrades;
      openTrades += item.openTrades;
      cumulativePosHoldTimeMs += item.cumulativePosHoldTime;

      if (item.isLongStrategy) {
        totalInLongStrategies += item.valueInOpenPositions;
      } else {
        totalInShortStrategies += item.valueInOpenPositions;
      }
    });

    return {
      symbols,
      unrealizedProfit,
      cumulativeValueInOpenPositions,
      meanTradeResPerc: safeDivide(
        cumulativeMeanTradeResPerc,
        totalCompletedTrades,
        0,
      ),
      realizedProfit,
      totalCompletedTrades,
      openTrades,
      meanPosTime: formatSecondsIntoTime(
        safeDivide(cumulativePosHoldTimeMs, totalCompletedTrades, 0) / 1000,
      ),
      totalInShortStrategies,
      totalInLongStrategies,
    };
  };

  const infoDict = getInfoDict();

  return (
    <div>
      <Heading size={"lg"}>Strategies</Heading>
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
                <StatLabel>Completed trades</StatLabel>
                <StatNumber>{infoDict.totalCompletedTrades}</StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Open trades</StatLabel>
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
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Value at risk</StatLabel>
                <StatNumber>
                  $
                  {roundNumberFloor(infoDict.cumulativeValueInOpenPositions, 2)}
                </StatNumber>
              </Stat>
            </div>
            <div>
              <Stat color={COLOR_CONTENT_PRIMARY}>
                <StatLabel>Value at risk (%)</StatLabel>
                <StatNumber>
                  {roundNumberFloor(
                    safeDivide(
                      infoDict.cumulativeValueInOpenPositions,
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
                <StatLabel>Longs (%)</StatLabel>
                <StatNumber>
                  {roundNumberFloor(
                    safeDivide(
                      infoDict.totalInLongStrategies,
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
                <StatLabel>Shorts (%)</StatLabel>
                <StatNumber>
                  {roundNumberFloor(
                    safeDivide(
                      infoDict.totalInShortStrategies,
                      latestSnapshotQuery.data.value,
                      0,
                    ) * 100,
                    2,
                  )}
                  %
                </StatNumber>
              </Stat>
            </div>
          </div>
        </ChakraCard>
      </div>
      <div
        className="ag-theme-alpine-dark"
        style={{
          width: "100%",
          height: "calc(100vh - 330px)",
          marginTop: "16px",
        }}
      >
        <AgGridReact
          pagination={true}
          columnDefs={COLUMN_DEFS as any}
          paginationAutoPageSize={true}
          rowData={getDirectionalStrategies()}
        />
      </div>
    </div>
  );
};

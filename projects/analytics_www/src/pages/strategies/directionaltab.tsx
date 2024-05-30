import { Heading, Spinner, Text } from "@chakra-ui/react";
import { ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import {
  BinanceSymbolPrice,
  findCurrentPrice,
  getNumberDisplayColor,
  roundNumberFloor,
  StrategiesResponse,
  Strategy,
  StrategyGroup,
  Trade,
} from "common_js";
import { Link } from "react-router-dom";
import { useBinanceSpotPriceInfo } from "src/http/queries";
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

const profitColumnCellRenderer = (params: ICellRendererParams) => {
  return (
    <Text color={getNumberDisplayColor(params.value, COLOR_CONTENT_PRIMARY)}>
      {roundNumberFloor(params.value, 2)}$
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

        if (item.close_price && item.net_result && item.percent_result) {
          netResultTrades += item.net_result;
          cumulativePercResult += item.percent_result;
          completedTrades += 1;
        }

        if (item.open_price && !item.close_price) {
          openTrades += 1;
        }

        if (latestPrice && !item.close_price) {
          const unrealizedProfit = isLongStrategy
            ? (latestPrice - item.open_price) * item.quantity
            : (item.open_price - latestPrice) * item.quantity;
          cumulativeUnrealizedProfit += unrealizedProfit;
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
  };
};

export const DirectionalStrategiesTab = ({
  strategiesRes,
}: {
  strategiesRes: StrategiesResponse | undefined;
}) => {
  const binancePriceQuery = useBinanceSpotPriceInfo();
  if (!strategiesRes || !binancePriceQuery.data) {
    return <Spinner />;
  }

  const getDirectionalStrategies = (): any => {
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

  return (
    <div>
      <Heading size={"lg"}>Strategies</Heading>
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
          rowData={getDirectionalStrategies()}
        />
      </div>
    </div>
  );
};

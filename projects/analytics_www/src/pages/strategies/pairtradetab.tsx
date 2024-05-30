import { Heading, Spinner } from "@chakra-ui/react";
import { ColDef, ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import {
  LongShortGroup,
  LongShortTicker,
  StrategiesResponse,
  Trade,
} from "common_js";
import { Link } from "react-router-dom";
import { getLsStrategyPath } from "src/utils";

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
    headerName: "Net result",
    field: "netResult",
    sortable: true,
    editable: false,
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
) => {
  let numSymbols = 0;
  let openTrades = 0;
  let completedTrades = 0;
  let cumulativeNetResult = 0;

  longshortTickers.forEach((item) => {
    if (item.long_short_group_id === strategy.id) {
      numSymbols += 1;
    }
  });

  trades.forEach((item) => {
    if (item.pair_trade_group_id === strategy.id) {
      if (item.close_price && item.net_result && item.percent_result) {
        completedTrades += 1;
        cumulativeNetResult += item.net_result;
      }

      if (!item.close_price) {
        openTrades += 1;
      }
    }
  });

  return {
    size: numSymbols,
    openTrades,
    closedTrades: completedTrades,
    netResult: cumulativeNetResult,
    meanTradeResultPerc:
      completedTrades !== 0 ? cumulativeNetResult / completedTrades : 0,
  };
};

interface TableRow {
  size: number;
  openTrades: number;
  closedTrades: number;
  netResult: number;
  meanTradeResultPerc: number;
}

export const PairTradeTab = ({
  strategiesRes,
}: {
  strategiesRes: StrategiesResponse | undefined;
}) => {
  if (!strategiesRes) {
    return <Spinner />;
  }

  const getRows = () => {
    const ret: TableRow[] = [];
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
        ),
      };

      ret.push(lsStrategy);
    });

    return ret.sort((a, b) => b.netResult - a.netResult);
  };
  return (
    <div>
      <Heading size={"lg"}>Pair-trade strategies</Heading>
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

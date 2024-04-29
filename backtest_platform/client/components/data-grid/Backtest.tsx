import { AgGridReact } from "ag-grid-react";
import React, { useEffect, useState } from "react";
import { BacktestObject } from "../../clients/queries/response-types";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import "ag-grid-community/styles/ag-theme-alpine.css";
import "ag-grid-community/styles/ag-theme-balham.css";
import { ICellRendererParams } from "ag-grid-community";
import { ColDef } from "ag-grid-community";
import { usePathParams } from "../../hooks/usePathParams";
import { getDatasetBacktestPath } from "../../utils/navigate";
import { Link } from "react-router-dom";
import { roundNumberDropRemaining } from "../../utils/number";
import { Checkbox, UseDisclosureReturn } from "@chakra-ui/react";
import { useBacktestContext } from "../../context/backtest";
import { useMassBacktestContext } from "../../context/masspairtrade";
import { PATHS } from "../../utils/constants";

interface Props {
  backtests: BacktestObject[];
  onDeleteMode: UseDisclosureReturn;
}

type PathParams = {
  datasetName: string;
};

const checkboxCellRenderer = (params: ICellRendererParams) => {
  const { selectBacktest } = useBacktestContext();
  const { selectLongShortBacktest } = useMassBacktestContext();
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        height: "100%",
      }}
    >
      <Checkbox
        isChecked={params.value}
        onChange={() => {
          if (
            window.location.pathname.includes(PATHS.simulate.bulk_long_short)
          ) {
            selectLongShortBacktest(params.data.id);
          } else {
            selectBacktest(params.data.id);
          }
        }}
      />
    </div>
  );
};

const idCellRenderer = (params: ICellRendererParams) => {
  const { datasetName } = usePathParams<PathParams>();
  return (
    <Link
      to={getDatasetBacktestPath(datasetName, params.value)}
      className="link-default"
    >
      {params.value}
    </Link>
  );
};

const COLUMN_DEFS: ColDef[] = [
  {
    headerName: "Select",
    field: "select",
    sortable: true,
    editable: false,
    cellRenderer: checkboxCellRenderer,
  },
  {
    headerName: "ID",
    field: "id",
    sortable: true,
    editable: false,
    cellRenderer: idCellRenderer,
  },
  {
    headerName: "Name",
    field: "name",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Risk adj. ret. (%)",
    field: "risk_adjusted_return",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Result net",
    field: "result",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Result (%)",
    field: "result_perc",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Buy and hold result net",
    field: "buy_and_hold_result_net",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Buy and hold result (%)",
    field: "buy_and_hold_result_perc",
    sortable: true,
    editable: false,
  },
  {
    headerName: "CAGR (%)",
    field: "cagr",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Buy and hold CAGR (%)",
    field: "buy_and_hold_cagr",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Market exposure (%)",
    field: "market_exposure_time",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Max drawdown",
    field: "max_drawdown_perc",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Profit factor",
    field: "profit_factor",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Trade count",
    field: "trade_count",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Win %",
    field: "share_of_winning_trades_perc",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Loss %",
    field: "share_of_losing_trades_perc",
    sortable: true,
    editable: false,
  },
];

const createDatarowItems = (backtestObjects: BacktestObject[]) => {
  const ret = backtestObjects.map((item) => {
    return {
      id: item.id,
      name: item.name,
      result: roundNumberDropRemaining(
        item.end_balance - item.start_balance,
        0
      ),
      buy_and_hold_result_net: roundNumberDropRemaining(
        item.buy_and_hold_result_net,
        0
      ),
      buy_and_hold_result_perc: roundNumberDropRemaining(
        item.buy_and_hold_result_perc,
        2
      ),
      result_perc: roundNumberDropRemaining(item.result_perc, 2),
      profit_factor: roundNumberDropRemaining(item.profit_factor, 2),
      share_of_winning_trades_perc: roundNumberDropRemaining(
        item.share_of_winning_trades_perc,
        2
      ),
      share_of_losing_trades_perc: roundNumberDropRemaining(
        item.share_of_losing_trades_perc,
        2
      ),
      trade_count: item.trade_count,
      max_drawdown_perc: roundNumberDropRemaining(item.max_drawdown_perc, 2),
      cagr: roundNumberDropRemaining(item.cagr * 100, 2),
      buy_and_hold_cagr: roundNumberDropRemaining(
        item.buy_and_hold_cagr * 100,
        2
      ),
      market_exposure_time: roundNumberDropRemaining(
        item.market_exposure_time * 100,
        1
      ),
      risk_adjusted_return: roundNumberDropRemaining(
        item.risk_adjusted_return * 100,
        1
      ),
    };
  });
  return ret;
};

export const BacktestDatagrid = (props: Props) => {
  const { backtests, onDeleteMode } = props;

  const [rowData, setRowData] = useState(createDatarowItems(backtests));

  useEffect(() => {
    setRowData(createDatarowItems(backtests));
  }, [backtests]);

  return (
    <div
      className="ag-theme-alpine-dark"
      style={{ width: "100%", height: "calc(100vh - 170px)" }}
    >
      <AgGridReact
        pagination={true}
        columnDefs={COLUMN_DEFS.filter((item) => {
          if (!onDeleteMode.isOpen) {
            return item.headerName !== "Select";
          }
          return true;
        })}
        paginationAutoPageSize={true}
        rowData={rowData}
      />
    </div>
  );
};

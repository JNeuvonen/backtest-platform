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

interface Props {
  backtests: BacktestObject[];
}

type PathParams = {
  datasetName: string;
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
    };
  });
  return ret;
};

export const BacktestDatagrid = (props: Props) => {
  const { backtests } = props;

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
        columnDefs={COLUMN_DEFS}
        paginationAutoPageSize={true}
        rowData={rowData}
      />
    </div>
  );
};

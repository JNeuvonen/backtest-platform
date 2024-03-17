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
  { headerName: "Result", field: "result", sortable: true, editable: false },
];

const createDatarowItems = (backtestObjects: BacktestObject[]) => {
  const ret = backtestObjects.map((item) => {
    return {
      id: item.id,
      name: item.name,
      result: item.end_balance - item.start_balance,
    };
  });
  return ret;
};

export const BacktestDatagrid = (props: Props) => {
  const { backtests } = props;

  console.log(backtests);

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

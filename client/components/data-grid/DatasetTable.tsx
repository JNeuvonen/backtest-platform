import { CellClickedEvent, ColDef } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import "ag-grid-community/styles/ag-theme-alpine.css";
import "ag-grid-community/styles/ag-theme-balham.css";
import React, { useCallback, useEffect, useState } from "react";
import { fetchDatasetPagination } from "../../clients/requests";
import { convertColumnsToAgGridFormat } from "../../utils/dataset";

interface Props {
  columnDefs: ColDef[];
  columnLabels: string[];
  onCellClicked: (event: CellClickedEvent) => void;
  handleCellValueChanged: any;
  maxRows: number;
  datasetName: string;
}

export const DatasetTable = ({
  columnDefs,
  onCellClicked,
  handleCellValueChanged,
  maxRows,
  columnLabels,
  datasetName,
}: Props) => {
  const [gridApi, setGridApi] = useState<any>(null);
  const [stateColumnDefs] = useState(columnDefs);

  useEffect(() => {
    if (gridApi) {
      const dataSource = {
        getRows: (params: any) => {
          const pageSize = params.endRow - params.startRow;
          const page = params.endRow / pageSize;
          fetchDatasetPagination(datasetName, page, pageSize)
            .then((res) => {
              console.log(datasetName, page, pageSize);
              console.log(res.data);
              params.successCallback(
                convertColumnsToAgGridFormat(res.data, columnLabels),
                maxRows
              );
            })
            .catch(() => {});
        },
      };
      gridApi.setDatasource(dataSource);
    }
  }, [gridApi]);

  return (
    <div
      className="ag-theme-alpine-dark"
      style={{ width: "calc(100% + 32px)", height: "calc(100vh - 170px)" }}
    >
      <AgGridReact
        onGridReady={(params) => {
          setGridApi(params.api);
        }}
        columnDefs={stateColumnDefs}
        pagination={true}
        onColumnHeaderClicked={onCellClicked}
        onCellValueChanged={handleCellValueChanged}
        rowModelType={"infinite"}
        paginationAutoPageSize={true}
      />
    </div>
  );
};

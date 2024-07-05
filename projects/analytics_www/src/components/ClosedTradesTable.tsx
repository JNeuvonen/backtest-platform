import { ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import { CSSProperties } from "react";
import { Link } from "react-router-dom";
import {
  getDiffToPresentFormatted,
  getTradeCurrentProfitPerc,
  roundNumberFloor,
  Trade,
} from "src/common_js";
import { percColumnCellRenderer, profitColumnCellRenderer } from "src/pages";
import { getTradeViewPath } from "src/utils";

const idCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return null;
  }
  return (
    <Link to={getTradeViewPath(params.value)} className={"link-default"}>
      {params.value}
    </Link>
  );
};

const COLUMN_DEFS: any = [
  {
    headerName: "ID",
    field: "tradeId",
    sortable: true,
    editable: false,
    cellRenderer: idCellRenderer,
  },
  {
    headerName: "Symbol",
    field: "symbol",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Trade direction",
    field: "direction",
    sortable: false,
    editable: false,
  },
  {
    headerName: "Time open",
    field: "timeOpen",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Open price",
    field: "openPrice",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Close price",
    field: "closePrice",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Profit (%)",
    field: "profitPerc",
    sortable: true,
    editable: false,
    cellRenderer: percColumnCellRenderer,
  },
  {
    headerName: "Profit nominal",
    field: "profitNominal",
    sortable: true,
    editable: false,
    cellRenderer: profitColumnCellRenderer,
  },
];

interface Props {
  styles?: CSSProperties;
  trades: Trade[];
}

export const ClosedTradesTable = ({ styles, trades }: Props) => {
  const getRows = () => {
    const ret = [];

    trades.forEach((item) => {
      if (!item.close_price) {
        return;
      }
      const row = {
        tradeId: item.id,
        symbol: item.symbol,
        direction: item.direction,
        timeOpen: getDiffToPresentFormatted(new Date(item.created_at)),
        openPrice: roundNumberFloor(item.open_price, 5),
        closePrice: roundNumberFloor(item.close_price, 5),
        profitPerc: roundNumberFloor(
          getTradeCurrentProfitPerc(item, item.close_price),
          2,
        ),
        profitNominal: roundNumberFloor(item.net_result, 2),
      };
      ret.push(row);
    });

    return ret;
  };
  return (
    <div>
      <div
        className="ag-theme-alpine-dark"
        style={{
          width: "100%",
          marginTop: "16px",
          ...styles,
        }}
      >
        <AgGridReact
          columnDefs={COLUMN_DEFS as any}
          paginationAutoPageSize={true}
          rowData={getRows()}
          domLayout="autoHeight"
        />
      </div>
    </div>
  );
};

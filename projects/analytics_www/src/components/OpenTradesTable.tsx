import { AgGridReact } from "ag-grid-react";
import { CSSProperties } from "react";
import {
  findSymbolCurrentPrice,
  findSymbolPriceChange24h,
  getTradeCurrentProfitPerc,
  Trade,
} from "src/common_js";
import { useBinance24hPriceChanges, useBinanceSpotPriceInfo } from "src/http";
import { getDiffToPresentFormatted } from "src/common_js";
import { roundNumberFloor } from "src/common_js";
import { percColumnCellRenderer } from "src/pages";

const COLUMN_DEFS: any = [
  {
    headerName: "Symbol",
    field: "symbol",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Time open",
    field: "timeOpen",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Size nominal",
    field: "sizeNominal",
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
    headerName: "Current price",
    field: "currentPrice",
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
    headerName: "24h price change",
    field: "priceChange24h",
    sortable: true,
    editable: false,
  },
];

interface Props {
  styles?: CSSProperties;
  trades: Trade[];
}

export const OpenTradesTable = ({ styles, trades }: Props) => {
  const binancePrices = useBinanceSpotPriceInfo();
  const binancePriceChanges = useBinance24hPriceChanges();

  const getRows = () => {
    const ret = [];
    trades.forEach((item) => {
      const price = findSymbolCurrentPrice(
        item.symbol,
        binancePrices.data || [],
      );
      const row = {
        symbol: item.symbol,
        timeOpen: getDiffToPresentFormatted(new Date(item.created_at)),
        openPrice: roundNumberFloor(item.open_price, 5),
        currentPrice: roundNumberFloor(price, 5),
        sizeNominal: roundNumberFloor(price * item.quantity, 2),
        profitPerc: roundNumberFloor(getTradeCurrentProfitPerc(item, price), 2),
        priceChange24h: findSymbolPriceChange24h(
          item.symbol,
          binancePriceChanges.data || [],
        ),
      };
      ret.push(row);
    });
    return ret;
  };

  return (
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
  );
};

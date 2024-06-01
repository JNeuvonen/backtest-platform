import { Heading, Spinner } from "@chakra-ui/react";
import { AgGridReact } from "ag-grid-react";
import {
  ASSETS,
  findCurrentPrice,
  findSymbolPriceChangeTicker,
  roundNumberFloor,
  safeDivide,
} from "common_js";
import {
  useBinance24hPriceChanges,
  useBinanceAssets,
  useBinanceSpotPriceInfo,
  useLatestBalanceSnapshot,
} from "src/http/queries";
import {
  percColumnCellRenderer,
  profitColumnCellRenderer,
} from "../strategies";

const COLUMN_DEFS: any = [
  {
    headerName: "Asset",
    field: "asset",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Net asset nominal",
    field: "netAsset",
    sortable: true,
    editable: false,
    cellRenderer: profitColumnCellRenderer,
  },
  {
    headerName: "Net asset (%)",
    field: "netAssetPerc",
    sortable: true,
    editable: false,
    cellRenderer: percColumnCellRenderer,
  },
  {
    headerName: "Free (% of NAV)",
    field: "freeOfNav",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Debt (% of NAV)",
    field: "debtOfNav",
    sortable: true,
    editable: false,
  },
  {
    headerName: "24h price change",
    field: "priceChange24h",
    sortable: true,
    editable: false,
    cellRenderer: percColumnCellRenderer,
  },
];

export const AssetsPage = () => {
  const binanceAssets = useBinanceAssets();
  const binancePrices = useBinanceSpotPriceInfo();
  const latestBalanceSnapshot = useLatestBalanceSnapshot();
  const binancePriceChanges = useBinance24hPriceChanges();

  if (
    !binanceAssets.data ||
    !binancePrices.data ||
    !latestBalanceSnapshot.data
  ) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const getRows = () => {
    const ret = [];

    binanceAssets.data.forEach((item) => {
      const price = findCurrentPrice(
        item.asset + ASSETS.usdt,
        binancePrices.data,
      );

      const priceChange = findSymbolPriceChangeTicker(
        item.asset + ASSETS.usdt,
        binancePriceChanges.data,
      );

      if (!price || !priceChange) return;

      const value = item.netAsset * price;
      const freeValue = item.free * price;
      const debtValue = item.borrowed * price;

      ret.push({
        netAsset: value,
        netAssetPerc:
          safeDivide(value, latestBalanceSnapshot.data.value, 0) * 100,
        asset: item.asset,
        freeOfNav: roundNumberFloor(
          safeDivide(freeValue, latestBalanceSnapshot.data.value, 0) * 100,
          2,
        ),
        debtOfNav: roundNumberFloor(
          safeDivide(debtValue, latestBalanceSnapshot.data.value, 0) * 100,
          2,
        ),
        priceChange24h: roundNumberFloor(
          Number(priceChange.priceChangePercent),
          2,
        ),
      });
    });
    return ret.sort((a, b) => b.netAsset - a.netAsset);
  };

  return (
    <div>
      <div>
        <Heading size={"lg"}>Assets</Heading>
      </div>
      <div>
        <div
          className="ag-theme-alpine-dark"
          style={{
            width: "100%",
            marginTop: "16px",
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
    </div>
  );
};

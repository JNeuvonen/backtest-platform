import { Heading, Spinner } from "@chakra-ui/react";
import { AgGridReact } from "ag-grid-react";
import {
  ASSETS,
  findCurrentPrice,
  findNumOpenPositions,
  findSymbolPriceChangeTicker,
  includeLwrCase,
  roundNumberFloor,
  safeDivide,
} from "src/common_js";
import { useState } from "react";
import { ChakraInput } from "src/components/chakra";
import {
  useBinance24hPriceChanges,
  useBinanceAssets,
  useBinanceSpotPriceInfo,
  useLatestBalanceSnapshot,
  useUncompletedTradesQuery,
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
  {
    headerName: "Open trades",
    field: "openTrades",
    sortable: true,
    editable: false,
  },
];

export const AssetsPage = () => {
  const binanceAssets = useBinanceAssets();
  const binancePrices = useBinanceSpotPriceInfo();
  const latestBalanceSnapshot = useLatestBalanceSnapshot();
  const binancePriceChanges = useBinance24hPriceChanges();
  const uncompletedTrades = useUncompletedTradesQuery();

  const [assetFilterInput, setAssetFilterInput] = useState("");

  if (
    !binanceAssets.data ||
    !binancePrices.data ||
    !latestBalanceSnapshot.data ||
    !binancePriceChanges.data
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
      if (assetFilterInput && !includeLwrCase(item.asset, assetFilterInput)) {
        return;
      }

      const price = findCurrentPrice(
        item.asset + ASSETS.usdt,
        binancePrices.data,
      );

      const priceChange = findSymbolPriceChangeTicker(
        item.asset + ASSETS.usdt,
        binancePriceChanges.data,
      );

      if ((!price || !priceChange) && item.asset !== ASSETS.usdt) return;

      const value =
        item.asset !== ASSETS.usdt ? item.netAsset * price : item.netAsset;
      const freeValue =
        item.asset !== ASSETS.usdt ? item.free * price : item.free;
      const debtValue =
        item.asset !== ASSETS.usdt ? item.borrowed * price : item.borrowed;

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
        priceChange24h: priceChange
          ? roundNumberFloor(Number(priceChange.priceChangePercent), 2)
          : undefined,
        openTrades: findNumOpenPositions(
          item.asset,
          uncompletedTrades.data || [],
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
        <ChakraInput
          label="Filter asset"
          onChange={(val) => setAssetFilterInput(val)}
        />
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

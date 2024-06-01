import { Heading, MenuButton, MenuItem, Spinner } from "@chakra-ui/react";
import { AgGridReact } from "ag-grid-react";
import { findLongShortPair, getDiffToPresentFormatted } from "common_js";
import { useEffect, useState } from "react";
import { CgSync } from "react-icons/cg";
import { FaFilter } from "react-icons/fa";
import { IoMdAdd } from "react-icons/io";
import { ChakraMenu } from "src/components/chakra";
import { usePathParams } from "src/hooks";
import { useLongshortGroup } from "src/http/queries";

const COLUMN_DEFS = [
  {
    headerName: "Symbol",
    field: "symbol",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Is valid buy",
    field: "isValidBuy",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Is valid sell",
    field: "isValidSell",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Is no loan available",
    field: "isNoLoanAvailable",
    sortable: true,
    editable: false,
  },
  {
    headerName: "In position",
    field: "inPosition",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Currently pair buy",
    field: "isPairBuy",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Currently pair sell",
    field: "isPairSell",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Position time",
    field: "tradeOpened",
    sortable: true,
    editable: false,
  },
];

export const LongShortTickersPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const longshortGroupQuery = useLongshortGroup(strategyName);
  const [isRowDataSet, setIsRowDataSet] = useState(false);

  const [rowData, setRowData] = useState<any[]>([]);

  const getRows = () => {
    const ret = [];

    longshortGroupQuery.data.tickers.forEach((item) => {
      const pair = findLongShortPair(item.id, longshortGroupQuery.data.pairs);

      ret.push({
        symbol: item.symbol,
        isValidBuy: item.is_valid_buy,
        isValidSell: item.is_valid_sell,
        inPosition: pair ? true : false,
        isPairBuy: pair ? pair.buy_ticker_id === item.id : false,
        isPairSell: pair ? pair.sell_ticker_id === item.id : false,
        isNoLoanAvailable: pair ? pair.is_no_loan_available_err : false,
        tradeOpened: pair
          ? getDiffToPresentFormatted(new Date(pair.buy_open_time_ms))
          : undefined,
      });
    });
    return ret.sort((a, b) => Number(b.inPosition) - Number(a.inPosition));
  };

  useEffect(() => {
    if (!isRowDataSet && longshortGroupQuery.data) {
      const rows = getRows();
      setRowData(rows);
      setIsRowDataSet(true);
    }
  }, [longshortGroupQuery.data, isRowDataSet]);

  if (!longshortGroupQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }
  return (
    <div>
      <div>
        <div>
          <Heading size={"lg"}>
            {longshortGroupQuery.data.group.name.replace("{SYMBOL}_", "")}
          </Heading>
        </div>
        <div
          style={{
            marginTop: "8px",
            display: "flex",
            alignItems: "end",
            gap: "16px",
          }}
        >
          <ChakraMenu menuButton={<MenuButton>File</MenuButton>}>
            <MenuItem icon={<CgSync />} onClick={() => {}}>
              Save changes
            </MenuItem>
            <MenuItem icon={<FaFilter />} onClick={() => {}}>
              Filter
            </MenuItem>
            <MenuItem icon={<IoMdAdd />} onClick={() => {}}>
              Add tickers
            </MenuItem>
          </ChakraMenu>
        </div>
      </div>
      {rowData.length > 0 && (
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
              rowData={rowData}
              domLayout="autoHeight"
            />
          </div>
        </div>
      )}
    </div>
  );
};

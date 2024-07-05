import {
  Heading,
  MenuButton,
  MenuItem,
  Spinner,
  useDisclosure,
} from "@chakra-ui/react";
import { AgGridReact } from "ag-grid-react";
import { MdOutlineDataArray } from "react-icons/md";
import { CgSync } from "react-icons/cg";
import { FaFilter } from "react-icons/fa";
import _, { cloneDeep } from "lodash";
import {
  findCurrentPrice,
  getDiffToPresentInHours,
  roundNumberFloor,
  Strategy,
  Trade,
  TRADE_DIRECTIONS,
} from "src/common_js";
import { useEffect, useState } from "react";
import { ChakraInput, ChakraMenu } from "src/components/chakra";
import { useForceUpdate, usePathParams } from "src/hooks";
import {
  useBinanceSpotPriceInfo,
  useStrategyGroupQuery,
} from "src/http/queries";
import { ChakraDrawer } from "src/components/chakra/Drawer";
import { BulkUpdateRowsForm } from "./bulkupdaterowsform";
import { updateManyStrategies } from "src/http";
import { toast } from "react-toastify";
import {
  ProfitColumnCellRenderer,
  TradeIdCellRenderer,
} from "src/components/data-grid/cells";

const COLUMN_DEFS: any = [
  {
    headerName: "Symbol",
    field: "symbol",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Is enabled",
    field: "isEnabled",
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
    headerName: "View trade",
    field: "activeTradeId",
    sortable: true,
    editable: false,
    cellRenderer: TradeIdCellRenderer,
  },
  {
    headerName: "Trades",
    field: "tradesCount",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Profit",
    field: "realizedProfit",
    sortable: true,
    editable: false,
    cellRenderer: ProfitColumnCellRenderer,
  },
  {
    headerName: "Unrealized",
    field: "unrealizedProfit",
    sortable: true,
    editable: false,
    cellRenderer: ProfitColumnCellRenderer,
  },
  {
    headerName: "Value at risk",
    field: "valueAtRisk",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Allocation",
    field: "allocation",
    sortable: true,
    editable: true,
  },
  {
    headerName: "Close only",
    field: "closeOnly",
    sortable: true,
    editable: true,
  },
  {
    headerName: "Stop processing new candles",
    field: "stopProcessingNewCandles",
    sortable: true,
    editable: true,
  },
  {
    headerName: "Should enter",
    field: "shouldEnter",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Should close",
    field: "shouldClose",
    sortable: true,
    editable: true,
  },
  {
    headerName: "Position held",
    field: "positionHeldTime",
    sortable: true,
    editable: true,
  },
];

export const StrategySymbolsPage = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const strategyGroupQuery = useStrategyGroupQuery(strategyName);
  const [symbolFilterInput, setSymbolFilterInput] = useState("");
  const binanceSpotSymbols = useBinanceSpotPriceInfo();
  const [rowData, setRowData] = useState<any[]>([]);
  const [rowDataCopy, setRowDataCopy] = useState<any[]>([]);
  const [rowDataInitiated, setRowDataInitiated] = useState(false);
  const [isRowDataChanged, setIsRowDataChanged] = useState(false);
  const filterDrawer = useDisclosure();
  const bulkUpdateStrategies = useDisclosure();
  const forceUpdate = useForceUpdate();

  const getRowTradesDict = (strategy: Strategy, trades: Trade[]) => {
    if (!binanceSpotSymbols.data) return {};

    let realizedProfit = 0;
    let unrealizedProfit = 0;
    let tradesCount = 0;
    let valueAtRisk = 0;

    trades.forEach((item) => {
      if (item.strategy_id === strategy.id) {
        const currentPrice = findCurrentPrice(
          strategy.symbol,
          binanceSpotSymbols.data,
        );
        tradesCount += 1;

        if (!currentPrice) return;

        if (item.close_price) {
          if (item.direction === TRADE_DIRECTIONS.long) {
            realizedProfit +=
              (item.close_price - item.open_price) * item.quantity;
          } else {
            realizedProfit +=
              (item.open_price - item.close_price) * item.quantity;
          }
        } else {
          valueAtRisk += currentPrice * item.quantity;

          if (item.direction === TRADE_DIRECTIONS.long) {
            unrealizedProfit +=
              (currentPrice - item.open_price) * item.quantity;
          } else {
            unrealizedProfit +=
              (item.open_price - currentPrice) * item.quantity;
          }
        }
      }
    });

    return {
      realizedProfit,
      unrealizedProfit,
      tradesCount,
      valueAtRisk: roundNumberFloor(valueAtRisk, 2),
    };
  };

  const getSymbolRows = () => {
    if (!strategyGroupQuery.data) return [];

    const strategies = strategyGroupQuery.data.strategies;
    const trades = strategyGroupQuery.data.trades;
    const rows: any = [];

    strategies.forEach((item) => {
      if (symbolFilterInput) {
        if (
          !item.symbol.toLowerCase().includes(symbolFilterInput.toLowerCase())
        ) {
          return;
        }
      }
      rows.push({
        symbol: item.symbol,
        ...getRowTradesDict(item, trades),
        allocation: item.allocated_size_perc,
        closeOnly: item.is_in_close_only,
        shouldEnter: item.should_enter_trade,
        shouldClose: item.should_close_trade,
        activeTradeId: item.active_trade_id,
        isEnabled: !item.is_disabled,
        inPosition: item.is_in_position,
        positionHeldTime: getDiffToPresentInHours(
          new Date(item.time_on_trade_open_ms),
        ),
        stopProcessingNewCandles: item.stop_processing_new_candles
          ? true
          : false,
      });
    });

    return rows.sort(
      (a: any, b: any) => Number(b.inPosition) - Number(a.inPosition),
    );
  };

  const onCellValueChanged = (params: any) => {
    const updatedRowData = rowData.map((row) => {
      if (row.symbol === params.data.symbol) {
        return { ...row, ...params.data };
      }
      return row;
    });

    const updatedRowDataCopy = rowDataCopy.map((row) => {
      if (row.symbol === params.data.symbol) {
        return { ...row, ...params.data };
      }
      return row;
    });
    setRowData(updatedRowData);
    setRowDataCopy(cloneDeep(updatedRowDataCopy));
    setIsRowDataChanged(true);
  };

  useEffect(() => {
    if (
      binanceSpotSymbols.data &&
      !rowDataInitiated &&
      strategyGroupQuery.data &&
      !strategyGroupQuery.isLoading
    ) {
      const rows = getSymbolRows();
      setRowData(rows);
      setRowDataCopy(_.cloneDeep(rows));
      forceUpdate();

      if (rows.length > 0) {
        setRowDataInitiated(true);
      }
    }
  }, [strategyGroupQuery.data, binanceSpotSymbols.data]);

  const symbolInputFilterOnChange = (newValue: string) => {
    if (!newValue) {
      setRowData(_.cloneDeep(rowDataCopy));
    } else {
      const newRowData = rowDataCopy.filter((item) =>
        item.symbol.toLowerCase().includes(newValue.toLowerCase()),
      );
      setRowData(newRowData);
    }
    setSymbolFilterInput(newValue);
  };

  const findStrategyBasedOnSymbol = (symbol: string) => {
    let strategy = null;
    strategyGroupQuery.data.strategies.forEach((item) => {
      if (item.symbol === symbol) {
        strategy = item;
      }
    });
    return strategy;
  };

  const syncWithServer = async () => {
    const payload = [];

    rowDataCopy.forEach((item) => {
      const strategy = findStrategyBasedOnSymbol(item.symbol);
      const strategyInfo = {
        id: strategy.id,
        allocation_size_perc: item.allocation,
        should_close_trade: item.shouldClose,
        is_in_close_only: item.closeOnly,
        stop_processing_new_candles: item.stopProcessingNewCandles,
      };
      payload.push(strategyInfo);
    });

    const res = await updateManyStrategies({ strategies: payload });

    if (res.success) {
      toast.success("Updated strategies", { theme: "dark" });
    }
  };

  if (!strategyGroupQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  return (
    <>
      <ChakraDrawer {...filterDrawer} title={"Filter symbols"}>
        <div>
          <ChakraInput
            label="Filter symbols"
            onChange={(val) => {
              symbolInputFilterOnChange(val);
            }}
          />
        </div>
      </ChakraDrawer>
      <ChakraDrawer
        {...bulkUpdateStrategies}
        title={"Bulk update strategies"}
        drawerContentStyles={{ maxWidth: "40%" }}
      >
        <BulkUpdateRowsForm
          onSubmit={(values) => {
            const newRowData = rowDataCopy.map((item) => {
              return {
                ...item,
                closeOnly: values.closeOnly,
                shouldClose: values.shouldCloseTrade,
                allocation: values.allocationPerSymbol,
                stopProcessingNewCandles: values.stopProcessingNewCandles,
              };
            });

            setRowDataCopy(newRowData);
            setRowData(_.cloneDeep(newRowData));
            setIsRowDataChanged(true);
            forceUpdate();
          }}
          onClose={bulkUpdateStrategies.onClose}
        />
      </ChakraDrawer>
      <div>
        <div>
          <Heading size={"lg"}>
            {strategyGroupQuery.data.strategy_group.name}
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
            <MenuItem
              icon={<CgSync />}
              onClick={() => {
                syncWithServer();
              }}
              isDisabled={!isRowDataChanged}
            >
              Save changes
            </MenuItem>
            <MenuItem
              icon={<MdOutlineDataArray />}
              onClick={bulkUpdateStrategies.onOpen}
            >
              Run changes on every symbol
            </MenuItem>
            <MenuItem icon={<FaFilter />} onClick={filterDrawer.onOpen}>
              Filter
            </MenuItem>
          </ChakraMenu>
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
                onCellValueChanged={onCellValueChanged}
              />
            </div>
          </div>
        )}
      </div>
    </>
  );
};

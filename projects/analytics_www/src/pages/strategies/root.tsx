import { Heading, Spinner } from "@chakra-ui/react";
import { AgGridReact } from "ag-grid-react";
import { useStrategiesQuery } from "src/http/queries";
import { ChakraTabs } from "src/components/chakra/Tabs";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import "ag-grid-community/styles/ag-theme-alpine.css";
import "ag-grid-community/styles/ag-theme-balham.css";
import {
  roundNumberFloor,
  StrategiesResponse,
  Strategy,
  StrategyGroup,
  Trade,
} from "common_js";

const COLUMN_DEFS = [
  {
    headerName: "Strategy name",
    field: "name",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Universe size",
    field: "size",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Open trades",
    field: "openTrades",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Net result",
    field: "netResult",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Num trades",
    field: "numTrades",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Mean trade (%)",
    field: "meanTradeResultPerc",
    sortable: true,
    editable: false,
  },
];

const getStrategyGroupFields = (
  strategyGroup: StrategyGroup,
  strategies: Strategy[],
  trades: Trade[],
) => {
  let universeSize = 0;
  let openTrades = 0;
  let netResultTrades = 0;
  let numTrades = 0;
  let cumulativePercResult = 0;
  let completedTrades = 0;

  const strategyIdToStrategyGroupIdMap: { [key: number]: number } = {};

  strategies.forEach((item) => {
    if (item.strategy_group_id === strategyGroup.id) {
      universeSize += 1;
    }

    if (item.strategy_group_id) {
      strategyIdToStrategyGroupIdMap[item.id] = item.strategy_group_id;
    }
  });

  trades.forEach((item) => {
    if (item.strategy_id) {
      const strategyGroupId = strategyIdToStrategyGroupIdMap[item.strategy_id];

      if (strategyGroupId === strategyGroup.id) {
        numTrades += 1;

        if (item.close_price && item.net_result && item.percent_result) {
          netResultTrades += item.net_result;
          cumulativePercResult += item.percent_result;
          completedTrades += 1;
        }

        if (item.open_price && !item.close_price) {
          openTrades += 1;
        }
      }
    }
  });

  return {
    size: universeSize,
    numTrades,
    netResult: roundNumberFloor(netResultTrades, 2),
    openTrades,
    meanTradeResultPerc:
      completedTrades !== 0
        ? roundNumberFloor(cumulativePercResult / completedTrades, 2)
        : 0,
  };
};

const DirectionalStrategies = ({
  strategiesRes,
}: {
  strategiesRes: StrategiesResponse;
}) => {
  const getDirectionalStrategies = (): any => {
    const ret = strategiesRes.strategy_groups;

    return ret.map((item: StrategyGroup) => {
      return {
        name: item.name,
        ...getStrategyGroupFields(
          item,
          strategiesRes.directional_strategies,
          strategiesRes.trades,
        ),
      };
    });
  };

  return (
    <div>
      <Heading size={"lg"}>Strategies</Heading>
      <div
        className="ag-theme-alpine-dark"
        style={{
          width: "100%",
          height: "calc(100vh - 170px)",
          marginTop: "16px",
        }}
      >
        <AgGridReact
          pagination={true}
          columnDefs={COLUMN_DEFS}
          paginationAutoPageSize={true}
          rowData={getDirectionalStrategies()}
        />
      </div>
    </div>
  );
};

export const StrategiesPage = () => {
  const strategiesQuery = useStrategiesQuery();

  if (strategiesQuery.isLoading || !strategiesQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }
  return (
    <div>
      <div>
        <ChakraTabs
          labels={["Directional", "Pair-trade"]}
          tabs={[
            <DirectionalStrategies strategiesRes={strategiesQuery.data} />,
            <div>Pair trade</div>,
          ]}
          top={16}
          style={{ overflowY: "scroll" }}
        />
      </div>
    </div>
  );
};

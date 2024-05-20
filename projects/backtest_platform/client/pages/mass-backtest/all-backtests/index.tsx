import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import {
  useManyBacktests,
  useMassbacktests,
} from "../../../clients/queries/queries";
import { Heading, Spinner } from "@chakra-ui/react";
import { AgGridReact } from "ag-grid-react";
import { ColDef, ICellRendererParams } from "ag-grid-community";
import { Link } from "react-router-dom";
import { getInvidualMassBacktestPath } from "../../../utils/navigate";
import {
  BacktestObject,
  MassBacktest,
} from "../../../clients/queries/response-types";
import { calculateMean, roundNumberDropRemaining } from "../../../utils/number";

interface PathParams {
  backtestId: number;
}

const idCellRenderer = (params: ICellRendererParams) => {
  return (
    <Link
      to={getInvidualMassBacktestPath(params.value)}
      className="link-default"
    >
      {params.value}
    </Link>
  );
};

const createRowDataAggregates = (
  massBacktests: MassBacktest,
  backtests: BacktestObject[]
) => {
  const backtestsFiltered = backtests.filter((item) =>
    massBacktests.backtest_ids.includes(item.id)
  );

  return {
    profit_factor: roundNumberDropRemaining(
      calculateMean(backtestsFiltered, "profit_factor"),
      2
    ),
    result_perc: roundNumberDropRemaining(
      calculateMean(backtestsFiltered, "result_perc"),
      2
    ),
    trade_count: roundNumberDropRemaining(
      calculateMean(backtestsFiltered, "trade_count"),
      0
    ),
    market_time_exposure: roundNumberDropRemaining(
      calculateMean(backtestsFiltered, "market_exposure_time") * 100,
      2
    ),
    cagr: roundNumberDropRemaining(
      calculateMean(backtestsFiltered, "cagr") * 100,
      2
    ),
    buy_and_hold_cagr: roundNumberDropRemaining(
      calculateMean(backtestsFiltered, "buy_and_hold_cagr") * 100,
      2
    ),
  };
};

const formatData = (
  massBacktests: MassBacktest[],
  backtests: BacktestObject[]
) => {
  return massBacktests.map((item) => {
    const rowAggregates = createRowDataAggregates(item, backtests);
    return {
      id: item.id,
      num_of_sims: item.backtest_ids.length,
      ...rowAggregates,
    };
  });
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
    headerName: "Num of sims",
    field: "num_of_sims",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Profit factor (mean)",
    field: "profit_factor",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Trade count (mean)",
    field: "trade_count",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Profit % (mean)",
    field: "result_perc",
    sortable: true,
    editable: false,
  },

  {
    headerName: "Time exp. % (mean)",
    field: "market_time_exposure",
    sortable: true,
    editable: false,
  },
  {
    headerName: "CAGR % (mean)",
    field: "cagr",
    sortable: true,
    editable: false,
  },
  {
    headerName: "Buy n Hold CAGR % (mean)",
    field: "buy_and_hold_cagr",
    sortable: true,
    editable: false,
  },
];

const MassBacktestsDatagrid = ({
  massBacktests,
}: {
  massBacktests: MassBacktest[];
}) => {
  const getAllIds = () => {
    const ret: number[] = [];
    massBacktests.forEach((item) => {
      item.backtest_ids.forEach((id) => ret.push(id));
    });
    return ret;
  };
  const useManyBacktestsQuery = useManyBacktests(getAllIds());

  if (
    useManyBacktestsQuery.isLoading ||
    !useManyBacktestsQuery.data ||
    !useManyBacktestsQuery.data.data
  ) {
    return (
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
          rowData={[]}
        />
      </div>
    );
  }
  return (
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
        rowData={formatData(massBacktests, useManyBacktestsQuery.data.data)}
      />
    </div>
  );
};

export const AllMassBacktests = () => {
  const { backtestId } = usePathParams<PathParams>();

  const massBacktestsQuery = useMassbacktests(Number(backtestId));

  if (massBacktestsQuery.isLoading || !massBacktestsQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  return (
    <div>
      <Heading size={"lg"}>Completed backtests on other symbols</Heading>

      <MassBacktestsDatagrid massBacktests={massBacktestsQuery.data} />
    </div>
  );
};

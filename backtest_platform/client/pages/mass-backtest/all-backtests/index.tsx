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
import { MassBacktest } from "../../../clients/queries/response-types";

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

const formatData = (massBacktests: MassBacktest[]) => {
  return massBacktests.map((item) => {
    return {
      id: item.id,
      num_of_sims: item.backtest_ids.length,
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
        rowData={formatData(massBacktests)}
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

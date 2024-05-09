import React from "react";
import { usePathParams } from "../../hooks/usePathParams";
import { useBacktestById } from "../../clients/queries/queries";
import { Heading, Spinner } from "@chakra-ui/react";
import ExternalLink from "../../components/ExternalLink";
import { getDatasetInfoPagePath } from "../../utils/navigate";
import { BacktestSummaryCard } from "../simulate/dataset/backtest/SummaryCard";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export const MLBasedBacktestPage = () => {
  const { backtestId, datasetName } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));

  if (!backtestQuery.data || !backtestQuery.data.data) return <Spinner />;

  const backtest = backtestQuery.data.data;

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          width: "100%",
          justifyContent: "space-between",
        }}
      >
        <Heading size={"lg"}>
          {datasetName} {!backtestQuery.data.data.name}
        </Heading>
        <div style={{ display: "flex", gap: "8px" }}>
          <ExternalLink
            to={getDatasetInfoPagePath(datasetName)}
            linkText={"Dataset"}
          />
        </div>
      </div>

      <BacktestSummaryCard backtest={backtest} />
    </div>
  );
};

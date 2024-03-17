import React from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useBacktestById } from "../../../../clients/queries/queries";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export const DatasetBacktestPage = () => {
  const { datasetName, backtestId } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  console.log(backtestQuery.data);
  return <div>Hello world</div>;
};

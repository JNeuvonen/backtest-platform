import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";

interface RouteParams {
  datasetName: string;
  modelName: string;
}

export const ModelInfoPage = () => {
  const { modelName, datasetName } = usePathParams<RouteParams>();
  return <div>model info page</div>;
};

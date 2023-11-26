import React from "react";
import { useParams } from "react-router-dom";

type DatasetDetailParams = {
  datasetName: string;
};

export const DatasetDetailPage = () => {
  const params = useParams<DatasetDetailParams>();
  const datasetName = params.datasetName || "";
  return <div>{datasetName}</div>;
};

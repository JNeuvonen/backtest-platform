import React from "react";
import { useColumnQuery } from "../clients/queries/queries";

interface Props {
  datasetName: string;
  columnName: string;
}

export const ColumnInfo = ({ datasetName, columnName }: Props) => {
  const { data, isLoading, isFetching, refetch } = useColumnQuery(
    datasetName,
    columnName
  );
  return <div>col info here</div>;
};

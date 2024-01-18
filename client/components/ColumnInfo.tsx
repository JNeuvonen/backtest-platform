import React from "react";
import { useColumnQuery } from "../clients/queries/queries";
import { Link } from "react-router-dom";
import { getDatasetColumnInfo } from "../utils/navigate";

interface Props {
  datasetName: string;
  columnName: string;
}

export const ColumnInfo = ({ datasetName, columnName }: Props) => {
  const { data, isLoading, isFetching, refetch } = useColumnQuery(
    datasetName,
    columnName
  );

  return (
    <div>
      <Link
        className="link-default"
        to={getDatasetColumnInfo(datasetName, columnName)}
      >
        Detailed column page
      </Link>
    </div>
  );
};

import React from "react";
import { Link } from "react-router-dom";
import { getDatasetColumnInfoPath } from "../utils/navigate";

interface Props {
  datasetName: string;
  columnName: string;
}

export const ColumnInfo = ({ datasetName, columnName }: Props) => {
  return (
    <div>
      <Link
        className="link-default"
        to={getDatasetColumnInfoPath(datasetName, columnName)}
      >
        Detailed column page
      </Link>
    </div>
  );
};

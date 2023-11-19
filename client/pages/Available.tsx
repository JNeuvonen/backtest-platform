import React from "react";
import { useDatasetsQuery } from "../clients/queries";
import { Spinner } from "@chakra-ui/react";
import { DatasetTable } from "../components/tables/Dataset";

export const AvailablePage = () => {
  const { data, isLoading } = useDatasetsQuery();

  if (isLoading) {
    return <Spinner />;
  }

  const renderDatasetsContainer = () => {
    if (isLoading) {
      return (
        <div>
          <Spinner />;
        </div>
      );
    }

    if (!data || !data?.res.tables) {
      return null;
    }

    return (
      <div>
        <DatasetTable tables={data.res.tables} />
      </div>
    );
  };
  return (
    <div>
      <h1>Available datasets</h1>
      {renderDatasetsContainer()}
    </div>
  );
};

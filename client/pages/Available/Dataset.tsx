import React from "react";
import { useParams } from "react-router-dom";
import { useDatasetQuery } from "../../clients/queries/queries";
import { Box, Spinner, Table, Th, Thead, Tr } from "@chakra-ui/react";
import { GenericTable } from "../../components/tables/GenericTable";

type DatasetDetailParams = {
  datasetName: string;
};

export const DatasetDetailPage = () => {
  const params = useParams<DatasetDetailParams>();
  const datasetName = params.datasetName || "";
  const { data, isLoading } = useDatasetQuery(datasetName);

  const columnOnClickFunction = (selectedColumn: string) => {};

  if (isLoading) {
    return <Spinner />;
  }

  const dataset = data?.res.dataset;

  if (!dataset) {
    return <Box>Page is not available</Box>;
  }

  const columns = dataset.columns;

  return (
    <div>
      <Box>{datasetName}</Box>
      <Box marginTop={"16px"}>
        <GenericTable
          columns={columns}
          rows={dataset.head}
          columnOnClickFunc={columnOnClickFunction}
        />
      </Box>
    </div>
  );
};

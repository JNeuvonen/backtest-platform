import React, { useState } from "react";
import { useParams } from "react-router-dom";
import { useDatasetQuery } from "../../clients/queries/queries";
import { Box, Button, Spinner } from "@chakra-ui/react";
import { GenericTable } from "../../components/tables/GenericTable";
import { ChakraModal } from "../../components/chakra/modal";
import { useModal } from "../../hooks/useOpen";
import { BUTTON_VARIANTS } from "../../theme";
import { ColumnModal } from "./ColumnModal";

type DatasetDetailParams = {
  datasetName: string;
};

export const DatasetDetailPage = () => {
  const params = useParams<DatasetDetailParams>();
  const datasetName = params.datasetName || "";
  const { isOpen, modalClose, setIsOpen } = useModal(false);
  const { data, isLoading } = useDatasetQuery(datasetName);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [showHead, setShowHead] = useState(false);
  const columnOnClickFunction = (selectedColumn: string) => {
    setSelectedColumn(selectedColumn);
    setIsOpen(true);
  };

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
      <ChakraModal
        isOpen={isOpen}
        title={`Column ${selectedColumn}`}
        onClose={modalClose}
        modalContentStyle={{
          minWidth: "max-content",
          maxWidth: "80%",
          marginTop: "10%",
        }}
      >
        <ColumnModal
          columnName={selectedColumn}
          datasetName={datasetName}
          close={modalClose}
        />
      </ChakraModal>
      <Box
        display={"flex"}
        justifyContent={"space-between"}
        alignItems={"center"}
      >
        <Box>{datasetName}</Box>
        <Box display={"flex"} gap={"16px"}>
          <Button variant={BUTTON_VARIANTS.grey}>Add dataset</Button>
          <Button variant={BUTTON_VARIANTS.grey}>Add columns</Button>
        </Box>
      </Box>
      <Box marginTop={"16px"}>
        <Box>Tail</Box>
        <GenericTable
          columns={columns}
          rows={dataset.tail}
          columnOnClickFunc={columnOnClickFunction}
        />
      </Box>
      <Box marginTop={"16px"}>
        <Button
          onClick={() => setShowHead(!showHead)}
          variant={BUTTON_VARIANTS.nofill}
        >
          {showHead ? "Hide head" : "Show head"}
        </Button>
        {showHead && (
          <GenericTable
            columns={columns}
            rows={dataset.head}
            columnOnClickFunc={columnOnClickFunction}
          />
        )}
      </Box>
    </div>
  );
};

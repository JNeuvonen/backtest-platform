import React, { useState } from "react";
import { useParams } from "react-router-dom";
import { useColumnQuery, useDatasetQuery } from "../../clients/queries/queries";
import { Box, Button, Spinner } from "@chakra-ui/react";
import { GenericTable } from "../../components/tables/GenericTable";
import { ChakraModal } from "../../components/chakra/modal";
import { useModal } from "../../hooks/useOpen";
import { Dataset } from "../../clients/queries/response-types";
import { ColumnChart } from "../../components/charts/column";
import { BUTTON_VARIANTS } from "../../theme";

type DatasetDetailParams = {
  datasetName: string;
};

interface ColumnModalContentProps {
  datasetName: string;
  columnName: string;
}

const ColumnModalContent = ({
  columnName,
  datasetName,
}: ColumnModalContentProps) => {
  const { data, isLoading } = useColumnQuery(datasetName, columnName);

  const massageDataForChart = (
    rows: number[][],
    kline_open_time: number[][]
  ) => {
    const itemCount = rows.length;
    const skipItems = Math.max(1, Math.floor(itemCount / 1000));
    const ret: Object[] = [];

    for (let i = 0; i < itemCount; i++) {
      if (i % skipItems === 0) {
        const item = rows[i];
        const rowObject = {};
        rowObject[columnName] = item[0];
        rowObject["kline_open_time"] = kline_open_time[i][0];
        console.log(rowObject);
        ret.push(rowObject);
      }
    }

    return ret;
  };

  if (isLoading) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const rows = data?.res.column.rows;
  const kline_open_time = data?.res.column.kline_open_time;
  if (!rows || !kline_open_time) return null;
  return (
    <div>
      <ColumnChart
        data={massageDataForChart(rows, kline_open_time)}
        xAxisDataKey={"kline_open_time"}
        lines={[{ dataKey: columnName, stroke: "red" }]}
      />
    </div>
  );
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
        modalContentStyle={{ minWidth: "max-content", maxWidth: "80%" }}
      >
        <ColumnModalContent
          columnName={selectedColumn}
          datasetName={datasetName}
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

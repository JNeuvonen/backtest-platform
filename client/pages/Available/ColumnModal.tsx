import React, { useState } from "react";
import { useColumnQuery } from "../../clients/queries/queries";
import { ColumnChart } from "../../components/charts/column";
import { Button, Input, Spinner, useToast } from "@chakra-ui/react";
import { useModal } from "../../hooks/useOpen";
import { ChakraModal } from "../../components/chakra/modal";
import { FormSubmitBar } from "../../components/form/CancelSubmitBar";
import { ConfirmModal } from "../../components/form/confirm";
import { renameColumnName } from "../../clients/requests";

interface ColumnModalContentProps {
  datasetName: string;
  columnName: string;
  close: () => void;
}

interface RenameColumnModalProps {
  datasetName: string;
  columnName: string;
  close: () => void;
}

const RenameColumnModal = ({
  datasetName,
  columnName,
  close,
}: RenameColumnModalProps) => {
  const toast = useToast();
  const [inputValue, setInputValue] = useState(columnName);
  const { isOpen, modalClose, setIsOpen } = useModal(false);

  const onSubmit = async () => {
    const res = await renameColumnName(datasetName, columnName, inputValue);

    if (res?.status === 200) {
      toast({
        title: `Renamed column to ${inputValue}`,
        status: "success",
        duration: 5000,
        isClosable: true,
      });
      close();
    } else {
      toast({
        title: "Failed to rename column",
        description: `Error: ${res?.status} - ${res?.res?.message}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      close();
    }
  };

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          setIsOpen(true);
        }}
      >
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
        />
        <FormSubmitBar
          cancelCallback={close}
          style={{ marginTop: "20px" }}
          submitDisabled={columnName === inputValue}
        />
      </form>

      <ConfirmModal
        isOpen={isOpen}
        onClose={modalClose}
        title="Confirm"
        message={
          <span>
            Are you sure you want to rename column <b>{columnName}</b> to{" "}
            <b>{inputValue}</b>?
          </span>
        }
        confirmText="Submit"
        cancelText="Cancel"
        onConfirm={onSubmit}
      />
    </div>
  );
};

export const ColumnModal = ({
  columnName,
  datasetName,
}: ColumnModalContentProps) => {
  const { data, isLoading } = useColumnQuery(datasetName, columnName);
  const {
    isOpen: renameIsOpen,
    modalClose: renameModalClose,
    setIsOpen: renameSetIsOpen,
  } = useModal(false);

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
    <>
      <ChakraModal
        isOpen={renameIsOpen}
        title={`Rename column ${columnName}`}
        onClose={renameModalClose}
        modalContentStyle={{ minWidth: "max-content", marginTop: "25%" }}
      >
        <RenameColumnModal
          columnName={columnName}
          close={renameModalClose}
          datasetName={datasetName}
        />
      </ChakraModal>

      <div>
        <div className="column-modal__toolbar">
          <Button onClick={() => renameSetIsOpen(true)}>Rename</Button>
          <Button>Edit</Button>
          <Button>Delete</Button>
        </div>
        <ColumnChart
          data={massageDataForChart(rows, kline_open_time)}
          xAxisDataKey={"kline_open_time"}
          lines={[{ dataKey: columnName, stroke: "red" }]}
        />
      </div>
    </>
  );
};

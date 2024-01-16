import React, { useEffect, useState } from "react";
import { useModal } from "../hooks/useOpen";
import { renameColumnName } from "../clients/requests";
import { dispatchDomEvent } from "../context/log";
import { DOM_EVENT_CHANNELS } from "../utils/constants";
import { Button, Input, Spinner, useToast } from "@chakra-ui/react";
import { FormSubmitBar } from "./form/CancelSubmitBar";
import { ConfirmModal } from "./form/confirm";
import { useColumnQuery } from "../clients/queries/queries";
import { URLS } from "../clients/endpoints";
import { ColumnChart } from "./charts/column";
import { ChakraModal } from "./chakra/modal";
import { ConfirmSwitch } from "./charts/confirm-switch";

interface ColumnModalContentProps {
  datasetName: string;
  columnName: string;
  close: () => void;
  setColumnName: React.Dispatch<React.SetStateAction<string>>;
}

interface RenameColumnModalProps {
  datasetName: string;
  columnName: string;
  close: () => void;
  setColumnName: React.Dispatch<React.SetStateAction<string>>;
  isTimeseriesCol: boolean;
}

const RenameColumnModal = ({
  datasetName,
  columnName,
  close,
  setColumnName,
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
      dispatchDomEvent({ channel: DOM_EVENT_CHANNELS.refetch_dataset });
    } else {
      toast({
        title: "Failed to rename column",
        description: `Error: ${res?.status} - ${res?.res?.message}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }

    setColumnName(inputValue);
    close();
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
  setColumnName,
}: ColumnModalContentProps) => {
  const { data, isLoading, isFetching, refetch } = useColumnQuery(
    datasetName,
    columnName
  );
  const [isTimeseriesCol, setIsTimeseriesCol] = useState<boolean>(false);
  const toast = useToast();

  useEffect(() => {
    if (data?.res.timeseries_col) {
      setIsTimeseriesCol(columnName === data.res.timeseries_col);
    }
  }, [data, columnName]);

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
    const ret: object[] = [];

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

  const changeTimeseriesColumn = async (newValue: boolean) => {
    setIsTimeseriesCol(newValue);

    const url = URLS.set_time_column(datasetName);
    const request = fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        new_timeseries_col: newValue ? columnName : null,
      }),
      method: "PUT",
    });

    request
      .then((res) => {
        if (res.status === 200) {
          toast({
            title: "Initiated background download on all of the pairs",
            status: "success",
            duration: 5000,
            isClosable: true,
          });
          refetch();
        } else {
          toast({
            title: "Error",
            description: "Changing the time series column was not succesful",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
        }
      })
      .catch((error) => {
        toast({
          title: "Error",
          description: error?.message,
          status: "error",
          duration: 5000,
          isClosable: true,
        });
      });
  };

  if (isLoading || isFetching) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const rows = data?.res.column?.rows;
  const kline_open_time = data?.res.column?.kline_open_time;
  if (!rows) return null;

  const getColumnChart = () => {
    if (!kline_open_time) {
      return (
        <div>
          Currently, there is no selected time column for the dataset. Please
          select a time column in order for charts to be visible.
        </div>
      );
    }
    return (
      <ColumnChart
        data={massageDataForChart(rows, kline_open_time)}
        xAxisDataKey={"kline_open_time"}
        lines={[{ dataKey: columnName, stroke: "red" }]}
      />
    );
  };

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
          setColumnName={setColumnName}
          isTimeseriesCol={isTimeseriesCol}
        />
      </ChakraModal>

      <div>
        <div className="column-modal__toolbar">
          <Button onClick={() => renameSetIsOpen(true)}>Rename</Button>
          <Button>Edit</Button>
          <Button>Delete</Button>
        </div>
        <div
          style={{
            marginBottom: "32px",
            marginTop: "16px",
            display: "flex",
            alignItems: "center",
            gap: "16px",
          }}
        >
          {
            <ConfirmSwitch
              isChecked={isTimeseriesCol}
              confirmTitle="Confirm"
              confirmText="Confirm"
              cancelText="Cancel"
              message={
                <span>
                  Are you sure you want to use the column <b>{columnName}</b> as
                  the datasets time series column?
                </span>
              }
              setNewToggleValue={changeTimeseriesColumn}
            />
          }
          <span>Use as the datasets time column</span>
        </div>
        {getColumnChart()}
      </div>
    </>
  );
};

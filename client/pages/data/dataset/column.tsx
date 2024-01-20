import React, { useState } from "react";
import Title from "../../../components/Title";
import { usePathParams } from "../../../hooks/usePathParams";
import { useColumnQuery } from "../../../clients/queries/queries";
import { SmallTable } from "../../../components/tables/Small";
import { Button, Spinner, useToast } from "@chakra-ui/react";
import { Column } from "../../../clients/queries/response-types";
import { makeUniDimensionalTableRows } from "../../../utils/table";
import { roundNumberDropRemaining } from "../../../utils/number";
import { BUTTON_VARIANTS } from "../../../theme";
import { PythonIcon } from "../../../components/icons/python";
import { ChakraModal } from "../../../components/chakra/modal";
import { useModal } from "../../../hooks/useOpen";
import { FormSubmitBar } from "../../../components/form/CancelSubmitBar";
import { createPythonCode } from "../../../utils/str";
import { ConfirmModal } from "../../../components/form/confirm";
import { execPythonOnDatasetCol } from "../../../clients/requests";
import { CODE } from "../../../utils/constants";
import { MdOutlineSubdirectoryArrowLeft } from "react-icons/md";
import { ToolBarStyle } from "../../../components/ToolbarStyle";
import { useNavigate } from "react-router-dom";
import { getDatasetEditorUrl } from "../../../utils/navigate";
import { CodeEditor } from "../../../components/CodeEditor";

interface RouteParams {
  datasetName: string;
  columnName: string;
}

interface ColumnsRes {
  timeseries_col: string | null;
  column: Column;
}

const COLUMNS_STATS_TABLE: string[] = [
  "Timeseries column",
  "Nulls",
  "Nr rows",
  "Max value",
  "Mean",
  "Median",
  "Min",
  "Std dev",
];

const { GET_DATASET_EXAMPLE, COL_SYMBOL } = CODE;

const getCodeDefaultValue = (columnName: string) => {
  return createPythonCode([
    `${GET_DATASET_EXAMPLE}`,
    `${COL_SYMBOL} = get_column() #${columnName}`,
    "",
    `#Multiply example. This multiplies the value of ${columnName} on every row.`,
    `#dataset[${COL_SYMBOL}] = dataset[${COL_SYMBOL}] * 3`,
    "",
    `#Create a new column example. This creates a new column based on the column ${columnName}.`,
    `#dataset["new_column"] = dataset[${COL_SYMBOL}]`,
    "",
    `#Dataset is a pandas dataframe. So all native pandas functions are available.`,
    `#This example creates a simple moving average of last 30 datapoints on the column ${columnName}`,
    `#dataset['SMA30_${columnName}'] = dataset['${columnName}'].rolling(window=30).mean()`,
  ]);
};

export const DatasetColumnInfoPage = () => {
  const { datasetName, columnName } = usePathParams<RouteParams>();
  const { data } = useColumnQuery(datasetName, columnName);
  const [code, setCode] = useState<string>(getCodeDefaultValue(columnName));
  const { isOpen, setIsOpen, modalClose } = useModal();
  const toast = useToast();
  const useSubmitConfirm = useModal();
  const navigate = useNavigate();

  if (!data?.res) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const res = data.res;

  const getStatsRows = (data: ColumnsRes) => {
    const rows: (string | number)[] = [];

    if (data.timeseries_col) {
      rows.push(data.timeseries_col);
    }
    rows.push(roundNumberDropRemaining(data.column.null_count, 3));
    rows.push(data.column.rows.length);

    if (data.column.stats) {
      rows.push(roundNumberDropRemaining(data.column.stats.max, 3));
      rows.push(roundNumberDropRemaining(data.column.stats.mean, 3));
      rows.push(roundNumberDropRemaining(data.column.stats.median, 3));
      rows.push(roundNumberDropRemaining(data.column.stats.min, 3));
      rows.push(roundNumberDropRemaining(data.column.stats.std_dev, 3));
    } else {
      rows.push("N/A");
      rows.push("N/A");
      rows.push("N/A");
      rows.push("N/A");
      rows.push("N/A");
    }
    return makeUniDimensionalTableRows(rows);
  };

  const onSubmit = async () => {
    const res = await execPythonOnDatasetCol(datasetName, columnName, code);

    if (res.status === 200) {
      toast({
        title: "Executed python code",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      useSubmitConfirm.modalClose();
    }
  };

  return (
    <>
      <ConfirmModal
        {...useSubmitConfirm}
        onClose={useSubmitConfirm.modalClose}
        onConfirm={onSubmit}
      />
      <div>
        <Title>Column {columnName}</Title>
        <ToolBarStyle style={{ marginTop: "16px" }}>
          <Button
            variant={BUTTON_VARIANTS.grey}
            leftIcon={<MdOutlineSubdirectoryArrowLeft />}
            onClick={() => navigate(getDatasetEditorUrl(datasetName))}
          >
            Return to dataset
          </Button>
          <div>
            <Button
              variant={BUTTON_VARIANTS.grey}
              leftIcon={<PythonIcon width={24} height={24} />}
              fontSize={14}
              onClick={() => setIsOpen(true)}
            >
              Python
            </Button>
          </div>
        </ToolBarStyle>
        <SmallTable
          columns={COLUMNS_STATS_TABLE}
          rows={getStatsRows(res)}
          containerStyles={{ maxWidth: "max-content", marginTop: "16px" }}
        />

        <ChakraModal
          isOpen={isOpen}
          title="Edit with python"
          onClose={modalClose}
          modalContentStyle={{
            minWidth: "max-content",
            minHeight: "80vh",
            maxWidth: "80vw",
            marginTop: "10vh",
            position: "relative",
          }}
          footerContent={
            <FormSubmitBar
              cancelCallback={modalClose}
              submitCallback={useSubmitConfirm.modalOpen}
            />
          }
        >
          <CodeEditor code={code} setCode={setCode} />
        </ChakraModal>
      </div>
    </>
  );
};

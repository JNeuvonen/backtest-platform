import React, { useState } from "react";
import Title from "../../components/Title";
import { usePathParams } from "../../hooks/usePathParams";
import { useColumnQuery } from "../../clients/queries/queries";
import { SmallTable } from "../../components/tables/Small";
import { Button, Spinner } from "@chakra-ui/react";
import { Column } from "../../clients/queries/response-types";
import { makeUniDimensionalTableRows } from "../../utils/table";
import { roundNumberDropRemaining } from "../../utils/number";
import { BUTTON_VARIANTS } from "../../theme";
import { PythonIcon } from "../../components/icons/python";
import { PythonEditor } from "../../components/PythonEditor";
import { OnMount } from "@monaco-editor/react";
import { CODE } from "../../utils/constants";
import { ChakraModal } from "../../components/chakra/modal";
import { useModal } from "../../hooks/useOpen";
import { FormSubmitBar } from "../../components/form/CancelSubmitBar";

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

const { INDENT } = CODE;

const getCodeDefaultValue = (columnName: string) => {
  return `def run_python(dataset):\n${INDENT}#dataset["${columnName}"] = dataset["${columnName}"] * 3\n${INDENT}#above line multiplies all values in the column '${columnName}' by 3`;
};

export const DatasetColumnInfoPage = () => {
  const { datasetName, columnName } = usePathParams<RouteParams>();
  const { data } = useColumnQuery(datasetName, columnName);
  const [code, setCode] = useState<string>(getCodeDefaultValue(columnName));
  const { isOpen, setIsOpen, modalClose } = useModal();

  if (!data?.res) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const handleCodeChange = (newValue: string | undefined) => {
    setCode(newValue ?? "");
  };

  const res = data.res;

  const getStatsRows = (data: ColumnsRes) => {
    const rows: (string | number)[] = [];

    if (data.timeseries_col) {
      rows.push(data.timeseries_col);
    }
    rows.push(roundNumberDropRemaining(data.column.null_count, 3));
    rows.push(data.column.rows.length);
    rows.push(roundNumberDropRemaining(data.column.stats.max, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.mean, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.median, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.min, 3));
    rows.push(roundNumberDropRemaining(data.column.stats.std_dev, 3));
    return makeUniDimensionalTableRows(rows);
  };

  const handleEditorDidMount: OnMount = (editor) => {
    editor.setValue(code);
    editor.setPosition({ lineNumber: 4, column: 5 });
    editor.focus();
  };

  const onSubmit = () => {};

  return (
    <div>
      <Title>Column {columnName}</Title>
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
            submitCallback={onSubmit}
          />
        }
      >
        <div
          style={{
            position: "relative",
            width: "100%",
            height: "100%",
            display: "flex",
          }}
        >
          <PythonEditor
            code={code}
            onChange={handleCodeChange}
            editorMount={handleEditorDidMount}
            height={"400px"}
            containerStyles={{ width: "65%", height: "100%" }}
            fontSize={15}
          />

          <div>Code presets will come here</div>
        </div>
      </ChakraModal>
    </div>
  );
};

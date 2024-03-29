import React, { useState } from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { useBacktestContext } from ".";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import { CREATE_COLUMNS_DEFAULT } from "../../utils/code";
import { useToast } from "@chakra-ui/react";
import { execPythonOnDataset } from "../../clients/requests";
import { CodeEditor } from "../../components/CodeEditor";
import { CODE_PRESET_CATEGORY } from "../../utils/constants";

export const RunPythonModal = () => {
  const toast = useToast();
  const { runPythonModal, datasetQuery, datasetName } = useBacktestContext();

  const [createColumnsCode, setCreateColumnsCode] = useState(
    CREATE_COLUMNS_DEFAULT()
  );

  const runPythonSubmit = async () => {
    const res = await execPythonOnDataset(
      datasetName,
      createColumnsCode,
      "NONE"
    );

    if (res.status === 200) {
      toast({
        title: "Executed python code",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      runPythonModal.onClose();
      datasetQuery.refetch();
      setCreateColumnsCode(CREATE_COLUMNS_DEFAULT());
    }
  };
  return (
    <ChakraModal
      {...runPythonModal}
      title="Run python"
      footerContent={
        <FormSubmitBar
          cancelCallback={runPythonModal.onClose}
          submitCallback={runPythonSubmit}
        />
      }
      modalContentStyle={{ maxWidth: "60%" }}
    >
      <CodeEditor
        code={createColumnsCode}
        setCode={setCreateColumnsCode}
        style={{ marginTop: "16px" }}
        fontSize={13}
        label="Create columns"
        disableCodePresets={true}
        codeContainerStyles={{ width: "100%" }}
        height={"250px"}
        presetCategory={CODE_PRESET_CATEGORY.backtest_create_columns}
      />
    </ChakraModal>
  );
};

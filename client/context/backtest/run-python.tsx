import React, { useState } from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { useBacktestContext } from ".";
import { FormSubmitBar } from "../../components/form/FormSubmitBar";
import { RunPythonOnAllCols } from "../../components/RunPythonOnAllCols";
import { CREATE_COLUMNS_DEFAULT } from "../../utils/code";
import { useToast } from "@chakra-ui/react";
import { execPythonOnDataset } from "../../clients/requests";

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
      isOpen={runPythonModal.isOpen}
      title={"Run python"}
      onClose={runPythonModal.onClose}
      modalContentStyle={{
        minWidth: "max-content",
        minHeight: "50%",
        maxWidth: "90%",
        marginTop: "10vh",
      }}
      footerContent={
        <FormSubmitBar
          cancelCallback={runPythonModal.onClose}
          submitCallback={runPythonSubmit}
        />
      }
    >
      <RunPythonOnAllCols
        code={createColumnsCode}
        setCode={setCreateColumnsCode}
      />
    </ChakraModal>
  );
};

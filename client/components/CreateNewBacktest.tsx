import React, { useState } from "react";
import { CodeEditor } from "./CodeEditor";
import { CREATE_COLUMNS_DEFAULT } from "../utils/code";
import { Spinner, Switch, useDisclosure, useToast } from "@chakra-ui/react";
import { WithLabel } from "./form/WithLabel";
import { ChakraModal } from "./chakra/modal";
import { Text } from "@chakra-ui/react";
import { TEXT_VARIANTS } from "../theme";
import { useDatasetQuery } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import { OverflopTooltip } from "./OverflowTooltip";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { execPythonOnDataset } from "../clients/requests";

type PathParams = {
  datasetName: string;
};

interface Props {
  enterTradeCode: string;
  exitTradeCode: string;
  setEnterTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setExitTradeCode: React.Dispatch<React.SetStateAction<string>>;
  doNotShort: boolean;
  setDoNotShort: React.Dispatch<React.SetStateAction<boolean>>;
}

export const CreateBacktestDrawer = (props: Props) => {
  const {
    enterTradeCode,
    exitTradeCode,
    setEnterTradeCode,
    setExitTradeCode,
    doNotShort,
    setDoNotShort,
  } = props;

  const { datasetName } = usePathParams<PathParams>();
  const [createColumnsCode, setCreateColumnsCode] = useState(
    CREATE_COLUMNS_DEFAULT()
  );
  const { refetch } = useDatasetQuery(datasetName);

  const columnsModal = useDisclosure();
  const runPythonModal = useDisclosure();

  const { data } = useDatasetQuery(datasetName);
  const toast = useToast();

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
      refetch();
      setCreateColumnsCode(CREATE_COLUMNS_DEFAULT());
    }
  };

  if (!data) return <Spinner />;

  return (
    <div>
      <ChakraModal {...columnsModal} title="Columns">
        <div id={"COLUMN_MODAL"}>
          {data.columns.map((item, idx) => {
            return (
              <div key={idx}>
                <OverflopTooltip text={item} containerId="COLUMN_MODAL">
                  <div>{item}</div>
                </OverflopTooltip>
              </div>
            );
          })}
        </div>
      </ChakraModal>

      <ChakraModal
        {...runPythonModal}
        title="Create columns"
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
        />
      </ChakraModal>

      <div style={{ display: "flex", gap: "16px" }}>
        <Text variant={TEXT_VARIANTS.clickable} onClick={columnsModal.onOpen}>
          Show columns
        </Text>

        <Text variant={TEXT_VARIANTS.clickable} onClick={runPythonModal.onOpen}>
          Create columns
        </Text>
      </div>
      <div>
        <CodeEditor
          code={enterTradeCode}
          setCode={setEnterTradeCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label="Enter trade criteria"
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
        />
      </div>

      <div style={{ marginTop: "32px" }}>
        <CodeEditor
          code={exitTradeCode}
          setCode={setExitTradeCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label="Exit trade criteria"
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
        />
      </div>
      <div style={{ marginTop: "16px" }}>
        <WithLabel label={"Do not use short selling"}>
          <Switch
            isChecked={doNotShort}
            onChange={() => setDoNotShort(!doNotShort)}
          />
        </WithLabel>
      </div>
    </div>
  );
};

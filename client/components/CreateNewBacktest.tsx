import React, { useState } from "react";
import { CodeEditor } from "./CodeEditor";
import { CREATE_COLUMNS_DEFAULT } from "../utils/code";
import {
  NumberInput,
  NumberInputField,
  Spinner,
  Switch,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { WithLabel } from "./form/WithLabel";
import { ChakraModal } from "./chakra/modal";
import { Text } from "@chakra-ui/react";
import { TEXT_VARIANTS } from "../theme";
import { useDatasetQuery } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import { OverflopTooltip } from "./OverflowTooltip";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { execPythonOnDataset } from "../clients/requests";
import { ChakraInput } from "./chakra/input";

type PathParams = {
  datasetName: string;
};

interface Props {
  openLongTradeCode: string;
  openShortTradeCode: string;
  closeLongTradeCode: string;
  closeShortTradeCode: string;

  setOpenLongTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setOpenShortTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setCloseLongTradeCode: React.Dispatch<React.SetStateAction<string>>;
  setCloseShortTradeCode: React.Dispatch<React.SetStateAction<string>>;

  useShorts: boolean;
  setUseShorts: React.Dispatch<React.SetStateAction<boolean>>;

  backtestName: string;
  setBacktestName: React.Dispatch<React.SetStateAction<string>>;

  klinesUntilClose: null | number;
  setKlinesUntilClose: React.Dispatch<React.SetStateAction<number | null>>;

  useTimeBasedClose: boolean;
  setUseTimeBasedClose: React.Dispatch<React.SetStateAction<boolean>>;
}

export const CreateBacktestDrawer = (props: Props) => {
  const {
    openLongTradeCode,
    openShortTradeCode,
    closeLongTradeCode,
    closeShortTradeCode,
    setOpenLongTradeCode,
    setOpenShortTradeCode,
    setCloseLongTradeCode,
    setCloseShortTradeCode,
    useShorts,
    setUseShorts,
    setBacktestName,
    useTimeBasedClose,
    setUseTimeBasedClose,
    klinesUntilClose,
    setKlinesUntilClose,
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

      <WithLabel>
        <ChakraInput label="Name (optional)" onChange={setBacktestName} />
      </WithLabel>
      <div>
        <CodeEditor
          code={openLongTradeCode}
          setCode={setOpenLongTradeCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label="Long condition"
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
        />
      </div>

      <div>
        <CodeEditor
          code={closeLongTradeCode}
          setCode={setCloseLongTradeCode}
          style={{ marginTop: "16px" }}
          fontSize={13}
          label="Close long condition"
          disableCodePresets={true}
          codeContainerStyles={{ width: "100%" }}
          height={"250px"}
        />
      </div>

      <div style={{ marginTop: "16px" }}>
        <WithLabel label={"Use short selling"}>
          <Switch
            isChecked={useShorts}
            onChange={() => setUseShorts(!useShorts)}
          />
        </WithLabel>
      </div>

      {useShorts && (
        <div style={{ marginTop: "16px" }}>
          <CodeEditor
            code={openShortTradeCode}
            setCode={setOpenShortTradeCode}
            style={{ marginTop: "16px" }}
            fontSize={13}
            label="Short condition"
            disableCodePresets={true}
            codeContainerStyles={{ width: "100%" }}
            height={"250px"}
          />
        </div>
      )}

      {useShorts && (
        <div style={{ marginTop: "16px" }}>
          <CodeEditor
            code={closeShortTradeCode}
            setCode={setCloseShortTradeCode}
            style={{ marginTop: "16px" }}
            fontSize={13}
            label="Close short condition"
            disableCodePresets={true}
            codeContainerStyles={{ width: "100%" }}
            height={"250px"}
          />
        </div>
      )}

      <div style={{ marginTop: "16px" }}>
        <WithLabel label={"Use time based closing strategy"}>
          <Switch
            isChecked={useTimeBasedClose}
            onChange={() => setUseTimeBasedClose(!useTimeBasedClose)}
          />
        </WithLabel>
      </div>

      {useTimeBasedClose && (
        <WithLabel
          label={"Klines until close"}
          containerStyles={{ maxWidth: "200px", marginTop: "16px" }}
        >
          <NumberInput
            step={5}
            min={0}
            value={klinesUntilClose || undefined}
            onChange={(valueString) =>
              setKlinesUntilClose(parseInt(valueString))
            }
          >
            <NumberInputField />
          </NumberInput>
        </WithLabel>
      )}
    </div>
  );
};

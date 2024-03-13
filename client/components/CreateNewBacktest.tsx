import React, { useState } from "react";
import { CodeEditor } from "./CodeEditor";
import { ENTER_TRADE_DEFAULT } from "../utils/code";
import { Spinner, Switch, useDisclosure } from "@chakra-ui/react";
import { WithLabel } from "./form/WithLabel";
import { ChakraModal } from "./chakra/modal";
import { Text } from "@chakra-ui/react";
import { TEXT_VARIANTS } from "../theme";
import { useDatasetQuery } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import { OverflopTooltip } from "./OverflowTooltip";

type PathParams = {
  datasetName: string;
};

export const CreateBacktestDrawer = () => {
  const { datasetName } = usePathParams<PathParams>();
  const [enterTradeCode, setEnterTradeCode] = useState(ENTER_TRADE_DEFAULT());
  const [exitTradeCode, setExitTradeCode] = useState(ENTER_TRADE_DEFAULT());
  const [doNotShort, setDoNotShort] = useState(true);
  const columnsModal = useDisclosure();
  const { data } = useDatasetQuery(datasetName);

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

      <div style={{ display: "flex", gap: "16px" }}>
        <Text variant={TEXT_VARIANTS.clickable} onClick={columnsModal.onOpen}>
          Show columns
        </Text>

        <Text variant={TEXT_VARIANTS.clickable} onClick={columnsModal.onOpen}>
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

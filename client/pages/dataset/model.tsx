import React, { useState } from "react";
import Title from "../../components/Title";
import { usePathParams } from "../../hooks/usePathParams";
import { useDatasetQuery } from "../../clients/queries/queries";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { SingleValue } from "react-select";
import {
  Button,
  Checkbox,
  RangeSlider,
  RangeSliderFilledTrack,
  RangeSliderThumb,
  RangeSliderTrack,
  Spinner,
} from "@chakra-ui/react";
import { ChakraSelect } from "../../components/chakra/select";
import {
  CodeHelper,
  DOM_IDS,
  NULL_FILL_STRATEGIES,
} from "../../utils/constants";
import { ToolBarStyle } from "../../components/ToolbarStyle";
import { CodeEditor } from "../../components/CodeEditor";
import { ChakraTooltip } from "../../components/Tooltip";
import { useAppContext } from "../../context/app";
import { KEYBIND_MSGS } from "../../utils/content";
import { ValidationSplitSlider } from "../../components/ValidationSplitSlider";
import { ChakraCheckbox } from "../../components/chakra/checkbox";

type PathParams = {
  datasetName: string;
};

const getModelDefaultExample = () => {
  const code = new CodeHelper();

  code.appendLine("class Model(nn.Module):");
  code.addIndent();

  code.appendLine("def __init__(self, n_input_params):");
  code.addIndent();

  code.appendLine("super(Model, self).__init__()");
  code.appendLine("self.linear = nn.Linear(n_input_params, 1)");
  code.appendLine("");

  code.reduceIndent();
  code.appendLine("def forward(self, x):");
  code.addIndent();
  code.appendLine("return self.linear(x)");

  return code.get();
};

const getHyperParamsExample = () => {
  const code = new CodeHelper();

  code.appendLine("criterion = nn.MSELoss()");
  code.appendLine(
    "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)"
  );

  return code.get();
};

export const DatasetModelPage = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data, isLoading, refetch } = useDatasetQuery(datasetName);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [modelCode, setModelCode] = useState(getModelDefaultExample());
  const [hyperParamsCode, setHyperParamsCode] = useState(
    getHyperParamsExample()
  );
  const [validSplitSize, setValidSplitSize] = useState([80, 100]);
  const [disableValSplit, setDisableValSplit] = useState(false);
  const { platform } = useAppContext();

  if (!data || !data?.res) {
    return (
      <div>
        <Title>Create Model</Title>
        <Spinner />
      </div>
    );
  }

  return (
    <div>
      <Title>Create Model</Title>

      <ToolBarStyle style={{ marginTop: "16px" }}>
        <ChakraTooltip label={KEYBIND_MSGS.get_save(platform)}>
          <Button style={{ height: "35px", marginBottom: "16px" }}>Save</Button>
        </ChakraTooltip>
      </ToolBarStyle>
      <ToolBarStyle>
        <SelectWithTextFilter
          containerStyle={{ width: "300px" }}
          label="Target column"
          options={data.res.dataset.columns.map((col) => {
            return {
              value: col,
              label: col,
            };
          })}
          isMulti={false}
          placeholder="Select column"
          onChange={(selectedOption) => {
            const option = selectedOption as SingleValue<OptionType>;
            setTargetColumn(option?.value as string);
          }}
        />
        <ChakraSelect
          containerStyle={{ width: "200px" }}
          label={"Null fill strategy"}
          options={NULL_FILL_STRATEGIES}
          id={DOM_IDS.select_null_fill_strat}
          defaultValueIndex={0}
        />
      </ToolBarStyle>
      <CodeEditor
        label="Model code"
        code={modelCode}
        setCode={setModelCode}
        style={{ marginTop: "16px" }}
        fontSize={13}
      />
      <CodeEditor
        code={hyperParamsCode}
        setCode={setHyperParamsCode}
        style={{ marginTop: "16px" }}
        fontSize={13}
        height="100px"
        label="Hyper parameters and optimizer"
      />

      <div style={{ marginTop: "16px" }}>
        <ChakraCheckbox
          label="Do not use validation split"
          isChecked={disableValSplit}
          onChange={() => setDisableValSplit(!disableValSplit)}
        />
        {!disableValSplit && (
          <ValidationSplitSlider
            sliderValue={validSplitSize}
            setSliderValue={setValidSplitSize}
            containerStyle={{ maxWidth: "300px", marginTop: "8px" }}
          />
        )}
      </div>
    </div>
  );
};

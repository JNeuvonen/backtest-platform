import React, { useEffect, useState } from "react";
import Title from "../../../components/typography/Title";
import { usePathParams } from "../../../hooks/usePathParams";
import { useDatasetQuery } from "../../../clients/queries/queries";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../../components/SelectFilter";
import { SingleValue } from "react-select";
import {
  Button,
  FormControl,
  FormLabel,
  Spinner,
  useToast,
  Text,
  Badge,
} from "@chakra-ui/react";
import { ChakraSelect } from "../../../components/chakra/select";
import {
  CodeHelper,
  DOM_IDS,
  NULL_FILL_STRATEGIES,
  NullFillStrategy,
} from "../../../utils/constants";
import { ToolBarStyle } from "../../../components/ToolbarStyle";
import { CodeEditor } from "../../../components/CodeEditor";
import { ValidationSplitSlider } from "../../../components/ValidationSplitSlider";
import { ChakraCheckbox } from "../../../components/chakra/checkbox";
import { BUTTON_VARIANTS, TEXT_VARIANTS } from "../../../theme";
import {
  CheckboxMulti,
  CheckboxValue,
} from "../../../components/form/CheckBoxMulti";
import { useForceUpdate } from "../../../hooks/useForceUpdate";
import { FormSubmitBar } from "../../../components/form/FormSubmitBar";
import { ChakraInput } from "../../../components/chakra/input";
import { createModel } from "../../../clients/requests";
import { nullFillStratToInt } from "../../../utils/navigate";
import { WithLabel } from "../../../components/form/WithLabel";

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
  code.appendLine("def get_criterion_and_optimizer(model):");
  code.addIndent();
  code.appendLine("criterion = nn.MSELoss()");
  code.appendLine(
    "optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)"
  );
  code.appendLine("return criterion, optimizer");
  return code.get();
};

interface Props {
  cancelCallback?: () => void;
  submitCallback: () => void;
}

export interface ModelDataPayload {
  name: string;
  target_col: string;
  drop_cols: string[];
  null_fill_strategy: number;
  model: string;
  hyper_params_and_optimizer_code: string;
  validation_split: number[];
}

export const DatasetModelCreatePage = ({
  cancelCallback,
  submitCallback,
}: Props) => {
  const { datasetName } = usePathParams<PathParams>();
  const { data } = useDatasetQuery(datasetName);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [columnsToDrop, setColumnsToDrop] = useState<CheckboxValue[]>([]);
  const [dropColumnsVisible, setDropColumnsVisible] = useState(false);
  const [modelName, setModelName] = useState("");
  const [nullFillStrategy, setNullFillStrategy] =
    useState<NullFillStrategy>("CLOSEST");
  const [modelCode, setModelCode] = useState(getModelDefaultExample());
  const [hyperParamsCode, setHyperParamsCode] = useState(
    getHyperParamsExample()
  );
  const [validSplitSize, setValidSplitSize] = useState([80, 100]);
  const [disableValSplit, setDisableValSplit] = useState(false);
  const forceUpdate = useForceUpdate();
  const toast = useToast();

  useEffect(() => {
    if (data?.status === 200) {
      setColumnsToDrop(
        data.res.dataset.columns.map((item) => {
          return {
            isChecked: false,
            label: item,
          };
        })
      );
    }
  }, [data]);

  if (!data || !data?.res) {
    return (
      <div>
        <Title>Create Model</Title>
        <Spinner />
      </div>
    );
  }

  const submit = async () => {
    const body: ModelDataPayload = {
      name: modelName,
      target_col: targetColumn,
      drop_cols: columnsToDrop
        .filter((item) => item.isChecked)
        .map((item) => item.label),
      null_fill_strategy: nullFillStratToInt(nullFillStrategy),
      model: modelCode,
      hyper_params_and_optimizer_code: hyperParamsCode,
      validation_split: validSplitSize,
    };

    const res = await createModel(datasetName, body);

    if (res.status === 200) {
      toast({
        title: "Created model",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      submitCallback();
    }
  };

  const submitIsDisabled = () => {
    return (
      !targetColumn || (validSplitSize[0] === 0 && validSplitSize[1] === 100)
    );
  };

  return (
    <div>
      <FormLabel>
        Target: <Badge colorScheme="green">{data.res.dataset.target_col}</Badge>
      </FormLabel>
      <ToolBarStyle style={{ marginTop: "16px" }}>
        <ChakraSelect
          containerStyle={{ width: "200px" }}
          label={"Null fill strategy"}
          options={NULL_FILL_STRATEGIES}
          id={DOM_IDS.select_null_fill_strat}
          defaultValueIndex={0}
          onChange={(value) => {
            setNullFillStrategy(value as NullFillStrategy);
          }}
        />
        <ChakraInput
          label="Model name"
          placeholder="Unique name"
          onChange={setModelName}
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

      <FormControl style={{ marginTop: "16px" }}>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={() => setDropColumnsVisible(!dropColumnsVisible)}
        >
          {dropColumnsVisible ? "Hide drop columns" : "Drop columns"}
        </Button>

        {dropColumnsVisible && (
          <CheckboxMulti
            options={columnsToDrop}
            onSelect={() => {
              forceUpdate();
            }}
          />
        )}
      </FormControl>

      <FormSubmitBar
        cancelCallback={cancelCallback}
        submitDisabled={submitIsDisabled()}
        submitCallback={submit}
      />
    </div>
  );
};

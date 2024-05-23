import React, { useEffect, useState } from "react";
import Title from "../../../components/typography/Title";
import { usePathParams } from "../../../hooks/usePathParams";
import { useDatasetQuery } from "../../../clients/queries/queries";
import {
  Button,
  FormControl,
  FormLabel,
  Spinner,
  useToast,
  Badge,
} from "@chakra-ui/react";
import { ChakraSelect } from "../../../components/chakra/Select";
import {
  CodeHelper,
  DOM_IDS,
  NULL_FILL_STRATEGIES,
  NullFillStrategy,
  SCALING_STRATEGIES,
  ScalingStrategy,
} from "../../../utils/constants";
import { ToolBarStyle } from "../../../components/ToolbarStyle";
import { CodeEditor } from "../../../components/CodeEditor";
import { ValidationSplitSlider } from "../../../components/ValidationSplitSlider";
import { ChakraCheckbox } from "../../../components/chakra/checkbox";
import { BUTTON_VARIANTS } from "../../../theme";
import {
  CheckboxMulti,
  CheckboxValue,
} from "../../../components/form/CheckBoxMulti";
import { useForceUpdate } from "../../../hooks/useForceUpdate";
import { FormSubmitBar } from "../../../components/form/FormSubmitBar";
import { createModel, setTargetColumn } from "../../../clients/requests";
import {
  nullFillStratToInt,
  scalingStrategyToInt,
} from "../../../utils/navigate";
import { ChakraPopover } from "../../../components/chakra/popover";
import { SelectColumnPopover } from "../../../components/SelectTargetColumnPopover";
import { useModal } from "../../../hooks/useOpen";
import { getDatasetColumnOptions } from "../../../utils/dataset";
import { ChakraInput } from "common_js";

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
  drop_cols: string[];
  null_fill_strategy: number;
  model: string;
  hyper_params_and_optimizer_code: string;
  validation_split: number[];
  scale_target: boolean;
  scaling_strategy: number;
  drop_cols_on_train: string[];
}

export const DatasetModelCreatePage = ({
  cancelCallback,
  submitCallback,
}: Props) => {
  const { datasetName } = usePathParams<PathParams>();
  const { data, refetch } = useDatasetQuery(datasetName);
  const [columnsToDrop, setColumnsToDrop] = useState<CheckboxValue[]>([]);
  const [dropColumnsVisible, setDropColumnsVisible] = useState(false);
  const [modelName, setModelName] = useState("");
  const [scaleTarget, setScaleTarget] = useState(false);
  const [nullFillStrategy, setNullFillStrategy] =
    useState<NullFillStrategy>("CLOSEST");
  const [scalingStrategy, setScalingStrategy] =
    useState<ScalingStrategy>("MIN-MAX");
  const [modelCode, setModelCode] = useState(getModelDefaultExample());
  const [hyperParamsCode, setHyperParamsCode] = useState(
    getHyperParamsExample()
  );
  const [validSplitSize, setValidSplitSize] = useState([80, 100]);
  const [disableValSplit, setDisableValSplit] = useState(false);
  const targetColumnPopover = useModal();
  const forceUpdate = useForceUpdate();
  const toast = useToast();

  useEffect(() => {
    if (data) {
      setColumnsToDrop(
        data.columns.map((item) => {
          return {
            isChecked: false,
            label: item,
          };
        })
      );
    }
  }, [data]);

  if (!data) {
    return (
      <div>
        <Title>Create Model</Title>
        <Spinner />
      </div>
    );
  }

  const submit = async () => {
    const droppedCols = columnsToDrop
      .filter((item) => item.isChecked)
      .map((item) => item.label);
    const body: ModelDataPayload = {
      name: modelName,
      drop_cols: columnsToDrop
        .filter((item) => item.isChecked)
        .map((item) => item.label),
      null_fill_strategy: nullFillStratToInt(nullFillStrategy),
      model: modelCode,
      hyper_params_and_optimizer_code: hyperParamsCode,
      validation_split: validSplitSize,
      scale_target: scaleTarget,
      scaling_strategy: scalingStrategyToInt(scalingStrategy),
      drop_cols_on_train: droppedCols,
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
      !data.target_col ||
      (validSplitSize[0] === 0 && validSplitSize[1] === 100) ||
      !modelName
    );
  };

  return (
    <div>
      {data.target_col ? (
        <FormLabel>
          Target: <Badge colorScheme="green">{data.target_col}</Badge>
        </FormLabel>
      ) : (
        <ChakraPopover
          isOpen={targetColumnPopover.isOpen}
          setOpen={targetColumnPopover.onOpen}
          onClose={targetColumnPopover.onClose}
          headerText="Set target column"
          body={
            <SelectColumnPopover
              options={getDatasetColumnOptions(data)}
              placeholder={data.target_col}
              selectCallback={(newCol) => {
                setTargetColumn(newCol, datasetName, () => {
                  toast({
                    title: "Changed target column",
                    status: "info",
                    duration: 5000,
                    isClosable: true,
                  });
                  targetColumnPopover.onClose();
                  refetch();
                });
              }}
            />
          }
        >
          <Button variant={BUTTON_VARIANTS.nofill}>Set target column</Button>
        </ChakraPopover>
      )}

      <ToolBarStyle style={{ marginTop: "16px" }}>
        <ChakraSelect
          containerStyle={{ width: "200px" }}
          label={"Scaling strategy"}
          options={SCALING_STRATEGIES}
          defaultValueIndex={0}
          onChange={(value) => {
            setScalingStrategy(value as ScalingStrategy);
          }}
        />
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
          label="Scale target"
          isChecked={scaleTarget}
          onChange={() => setScaleTarget(!scaleTarget)}
        />
      </div>

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
          {dropColumnsVisible
            ? "Hide drop columns"
            : "Drop columns on training"}
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

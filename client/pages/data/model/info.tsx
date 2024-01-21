import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useModelQuery } from "../../../clients/queries/queries";
import { Spinner } from "@chakra-ui/react";
import { CodeEditor } from "../../../components/CodeEditor";
import { ValidationSplitSlider } from "../../../components/ValidationSplitSlider";
import { SubTitle } from "../../../components/typography/SubTitle";

interface RouteParams {
  datasetName: string;
  modelName: string;
}

export const ModelInfoPage = () => {
  const { modelName } = usePathParams<RouteParams>();
  const { data } = useModelQuery(modelName);

  if (!data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  return (
    <div>
      <SubTitle>Model: {data.name}</SubTitle>
      <SubTitle>Target column: {data.target_col}</SubTitle>
      <CodeEditor
        code={data.model}
        readOnly={true}
        fontSize={15}
        disableCodePresets={true}
        codeContainerStyles={{ width: "100%", marginTop: "16px" }}
        label={"Model"}
      />
      <CodeEditor
        code={data.hyper_params_and_optimizer_code}
        readOnly={true}
        fontSize={15}
        disableCodePresets={true}
        codeContainerStyles={{ width: "100%", marginTop: "16px" }}
        label={"Hyper params and optimizer"}
        height="200px"
      />
      <ValidationSplitSlider
        sliderValue={data.validation_split}
        containerStyle={{ width: "250px", marginTop: "16px" }}
      />
    </div>
  );
};

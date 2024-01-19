import React, { useState } from "react";
import Title from "../../components/Title";
import { usePathParams } from "../../hooks/usePathParams";
import { useDatasetQuery } from "../../clients/queries/queries";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../components/SelectFilter";
import { SingleValue } from "react-select";
import { Spinner } from "@chakra-ui/react";
import { ChakraSelect } from "../../components/chakra/select";
import { DOM_IDS, NULL_FILL_STRATEGIES } from "../../utils/constants";
import { ToolBarStyle } from "../../components/ToolbarStyle";

type PathParams = {
  datasetName: string;
};

export const DatasetModelPage = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data, isLoading, refetch } = useDatasetQuery(datasetName);
  const [targetColumn, setTargetColumn] = useState<string>("");

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
    </div>
  );
};

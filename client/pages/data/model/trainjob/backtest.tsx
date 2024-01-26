import React from "react";
import {
  useDatasetsQuery,
  useTrainJobDetailed,
} from "../../../../clients/queries/queries";
import { usePathParams } from "../../../../hooks/usePathParams";
import { Button, Spinner } from "@chakra-ui/react";
import {
  ChakraSelect,
  SelectOption,
} from "../../../../components/chakra/select";
import {
  DatasetMetadata,
  DatasetResponse,
  EpochInfo,
} from "../../../../clients/queries/response-types";
import { roundNumberDropRemaining } from "../../../../utils/number";
import { Formik, Form, Field } from "formik";
import { CodeEditor } from "../../../../components/CodeEditor";
import { CodeHelper } from "../../../../utils/constants";
import {
  OptionType,
  SelectWithTextFilter,
} from "../../../../components/SelectFilter";
import { SingleValue } from "react-select";

interface BacktestForm {
  selectedModel: string | null;
}

const getTradeCriteriaDefaultCode = () => {
  const code = new CodeHelper();
  code.appendLine("def enter_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction > 1.01");
  code.appendLine("");
  code.reduceIndent();

  code.appendLine("def exit_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction < 0.99");
  return code.get();
};

export const BacktestModelPage = () => {
  const { trainJobId } = usePathParams<{
    trainJobId: string;
    datasetName?: string;
  }>();
  const { data } = useTrainJobDetailed(trainJobId);
  const { data: allDatasets } = useDatasetsQuery();
  console.log(allDatasets);

  if (!data || !allDatasets || !allDatasets.res) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const generateSelectWeightsOptions = (epochs: EpochInfo[]) => {
    const ret = [] as SelectOption[];
    for (let i = 0; i < epochs.length; i++) {
      const item = epochs[i];
      ret.push({
        label: `Epoch ${item.epoch}, Train loss: ${roundNumberDropRemaining(
          item.train_loss,
          3
        )}, Val: loss ${roundNumberDropRemaining(item.val_loss, 3)}`,
        value: String(item.epoch),
      });
    }
    return ret;
  };

  const generatePickColumnOptions = (tables: DatasetMetadata[]) => {
    const ret = [] as OptionType[];

    for (let i = 0; i < tables.length; i++) {
      const item = tables[i];

      const tableName = item.table_name;
      for (let j = 0; j < item.columns.length; j++) {
        const col = item.columns[j];
        ret.push({
          label: `${tableName}, ${col}`,
          value: `${tableName}, ${col}`,
        });
      }
    }
    return ret;
  };

  const handleSubmit = (values: BacktestForm) => {
    console.log(values);
  };

  return (
    <div>
      <Formik
        initialValues={{
          selectedModel: null,
          tradingCriteria: getTradeCriteriaDefaultCode(),
          priceColumn: null,
        }}
        onSubmit={handleSubmit}
      >
        {({ setFieldValue }) => (
          <Form>
            <Field name="selectedModel">
              {({ field }) => (
                <ChakraSelect
                  label="Select Model"
                  options={generateSelectWeightsOptions(data.epochs)}
                  onChange={(value) =>
                    field.onChange({ target: { name: "selectedModel", value } })
                  }
                  containerStyle={{ width: "350px" }}
                />
              )}
            </Field>

            <div style={{ marginTop: "16px" }}>
              <Field
                name="priceColumn"
                as={SelectWithTextFilter}
                options={generatePickColumnOptions(allDatasets.res.tables)}
                onChange={(selectedOptions: SingleValue<OptionType>) =>
                  setFieldValue("priceColumn", selectedOptions)
                }
                label="Select price column"
                containerStyle={{ width: "350px" }}
                isMulti={false}
                closeMenuOnSelect={false}
              />
            </div>

            <div style={{ marginTop: "16px" }}>
              <Field name="tradingCriteria">
                {({ field, form }) => {
                  return (
                    <CodeEditor
                      code={field.value}
                      setCode={(newCode) =>
                        form.setFieldValue("tradingCriteria", newCode)
                      }
                      label="Criteria for exit and enter"
                      fontSize={14}
                    />
                  );
                }}
              </Field>
            </div>

            <Button type="submit" marginTop={"16px"}>
              Submit
            </Button>
          </Form>
        )}
      </Formik>
    </div>
  );
};

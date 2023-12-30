import { AddIcon } from "@chakra-ui/icons";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Select,
  Spinner,
  Text,
  useToast,
} from "@chakra-ui/react";
import { Field, Form, Formik } from "formik";
import React, { useState } from "react";
import { MultiValue } from "react-select";
import { URLS } from "../clients/endpoints";
import { ApiResponse, buildRequest } from "../clients/fetch";
import {
  useBinanceTickersQuery,
  useDatasetsQuery,
} from "../clients/queries/queries";
import { BasicCard } from "../components/Card";
import { OptionType, SelectWithTextFilter } from "../components/SelectFilter";
import { ChakraModal } from "../components/chakra/modal";
import { DatasetTable } from "../components/tables/Dataset";
import { useModal } from "../hooks/useOpen";
import { DOM_EVENT_CHANNELS, GET_KLINE_OPTIONS } from "../utils/constants";
import { useMessageListener } from "../hooks/useMessageListener";
import { BinanceBasicTicker } from "../clients/queries/response-types";

const DATA_PROVIDERS = [
  {
    name: "Binance",
    icon: <AddIcon />,
  },
  {
    name: "Stocks",
    icon: <AddIcon />,
  },
];

const FormStateSelectProvider = ({
  advanceFormState,
}: {
  advanceFormState: (selectedProvider: string) => void;
}) => {
  return (
    <Box display={"flex"} marginTop={"32px"} gap={"32px"}>
      {DATA_PROVIDERS.map((item) => {
        return (
          <BasicCard
            key={item.name}
            p={12}
            onClick={() => advanceFormState(item.name)}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "16px",
            }}
          >
            <Text fontSize={"xx-large"}>{item.name} </Text>
            {item.icon}
          </BasicCard>
        );
      })}
    </Box>
  );
};

export const binanceTickSelectOptions = (tickers: BinanceBasicTicker[]) => {
  return tickers.map((item) => {
    return {
      value: item.symbol,
      label: item.symbol,
    };
  });
};

interface BinanceFormValues {
  selectedTickers: OptionType[];
  interval: string;
}

interface FormStateBinanceProps {
  modalClose: () => void;
}
const FormStateBinance = ({ modalClose }: FormStateBinanceProps) => {
  const toast = useToast();
  const { data, isLoading } = useBinanceTickersQuery();

  if (isLoading) {
    return <Spinner />;
  }

  if (!data || data?.status !== 200) {
    return <div>Binance API is not working currently</div>;
  }

  const submitForm = async (values: BinanceFormValues) => {
    const interval = values.interval;
    const symbols = values.selectedTickers;

    const promises: Promise<ApiResponse>[] = [];
    const url = URLS.binance_fetch_klines;

    symbols.map((item) => {
      const payload = {
        symbol: item.value,
        interval,
      };
      const req = buildRequest({
        method: "POST",
        url,
        payload,
      });
      promises.push(req);
    });

    try {
      await Promise.all(promises);
      toast({
        title: "Initiated background download on all of the pairs",
        status: "success",
        duration: 5000,
        isClosable: true,
      });
      modalClose();
    } catch (error) {
      toast({
        title: "Error",
        description: error?.message,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    }
  };
  return (
    <Formik
      initialValues={{ selectedTickers: [], interval: "1h" }}
      onSubmit={(values: BinanceFormValues, actions) => {
        actions.setSubmitting(true);
        submitForm(values);
        actions.setSubmitting(false);
      }}
    >
      {({ setFieldValue }) => (
        <Form>
          <Box>
            <FormControl>
              <FormLabel fontSize={"x-large"}>Pairs to be downloaded</FormLabel>
              <Field
                name="selectedTickers"
                as={SelectWithTextFilter}
                options={binanceTickSelectOptions(data.res.pairs)}
                onChange={(selectedOptions: MultiValue<OptionType>) =>
                  setFieldValue("selectedTickers", selectedOptions)
                }
                isMulti={true}
                closeMenuOnSelect={false}
              />
            </FormControl>

            <FormControl marginTop={"8px"}>
              <FormLabel fontSize={"x-large"}>Candle interval</FormLabel>
              <Field name="interval" as={Select}>
                {GET_KLINE_OPTIONS().map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </Field>
            </FormControl>

            <Button mt={4} type="submit">
              Submit
            </Button>
          </Box>
        </Form>
      )}
    </Formik>
  );
};
const FormStateStocks = () => {
  return <Box></Box>;
};

const STEP_1 = "select-provider";
const STEP_2 = "select-data";
interface GetNewDatasetModalProps {
  modalClose: () => void;
}
const GetNewDatasetModal = ({ modalClose }: GetNewDatasetModalProps) => {
  const [formState, setFormState] = useState(STEP_1);
  const [dataProvider, setDataProvider] = useState("");

  const advanceStepOne = (selectedProvider: string) => {
    setFormState(STEP_2);
    setDataProvider(selectedProvider);
  };

  return (
    <div style={{ width: "100%" }}>
      <div>
        {formState === STEP_1 && (
          <FormStateSelectProvider advanceFormState={advanceStepOne} />
        )}
        {formState === STEP_2 && dataProvider === "Binance" && (
          <FormStateBinance modalClose={modalClose} />
        )}
        {formState === STEP_2 && dataProvider === "Stocks" && (
          <FormStateStocks />
        )}
      </div>
    </div>
  );
};

export const AvailablePage = () => {
  const { data, isLoading, refetch } = useDatasetsQuery();
  const { isOpen, jsxContent, setContent, modalClose } = useModal(false);

  const refetchTables = () => {
    refetch();
  };

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_all_datasets,
    messageCallback: refetchTables,
  });

  if (isLoading) {
    return <Spinner />;
  }

  const renderDatasetsContainer = () => {
    if (isLoading) {
      return (
        <div>
          <Spinner />;
        </div>
      );
    }

    if (!data || !data?.res.tables) {
      return null;
    }

    return (
      <div>
        <DatasetTable tables={data.res.tables} />
      </div>
    );
  };
  return (
    <div>
      <ChakraModal
        isOpen={isOpen}
        title="New dataset"
        onClose={modalClose}
        modalContentStyle={{ maxWidth: "50%" }}
      >
        {jsxContent}
      </ChakraModal>
      <h1>Available datasets</h1>
      <Button
        onClick={() =>
          setContent(<GetNewDatasetModal modalClose={modalClose} />)
        }
      >
        Add from API
      </Button>
      {renderDatasetsContainer()}
    </div>
  );
};

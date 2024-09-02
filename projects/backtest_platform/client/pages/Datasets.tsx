import { AddIcon } from "@chakra-ui/icons";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Select,
  Spinner,
  Switch,
  Text,
  useToast,
} from "@chakra-ui/react";
import { Field, Form, Formik } from "formik";
import React, { useEffect, useState } from "react";
import { MultiValue } from "react-select";
import { LOCAL_API_URL } from "../clients/endpoints";
import { ApiResponse, buildRequest } from "../clients/fetch";
import {
  useBinanceTickersQuery,
  useDatasetsQuery,
  useNyseSymbolList,
} from "../clients/queries/queries";
import { BasicCard } from "../components/Card";
import { OptionType, SelectWithTextFilter } from "../components/SelectFilter";
import { ChakraModal } from "../components/chakra/modal";
import { DatasetTable } from "../components/tables/Dataset";
import { useModal } from "../hooks/useOpen";
import { DOM_EVENT_CHANNELS, GET_KLINE_OPTIONS } from "../utils/constants";
import { useMessageListener } from "../hooks/useMessageListener";
import { BinanceBasicTicker } from "../clients/queries/response-types";
import { useForceUpdate } from "../hooks/useForceUpdate";
import { BUTTON_VARIANTS } from "../theme";
import { removeDatasets, saveYfinanceKlines } from "../clients/requests";
import { ConfirmModal } from "../components/form/Confirm";
import { WithLabel } from "../components/form/WithLabel";
import { FormikDatePicker } from "../components/DatePicker";

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
  start_date: Date | null;
  end_date: Date | null;
}

interface FormStateBinanceProps {
  modalClose: () => void;
}
const FormStateBinance = ({ modalClose }: FormStateBinanceProps) => {
  const toast = useToast();
  const { data, isLoading } = useBinanceTickersQuery();
  const [useFutures, setUseFutures] = useState(false);

  if (isLoading) {
    return <Spinner />;
  }

  if (!data || data?.status !== 200) {
    return <div>Binance API is not working currently</div>;
  }

  const submitForm = async (values: BinanceFormValues) => {
    const { interval, start_date, end_date } = values;
    const symbols = values.selectedTickers;

    const promises: Promise<ApiResponse>[] = [];
    const url = LOCAL_API_URL.binance_fetch_klines;

    symbols.map((item) => {
      const payload = {
        symbol: item.value,
        interval,
        use_futures: useFutures,
        start_date_iso: start_date ? start_date.toISOString() : null,
        end_date_iso: end_date ? end_date.toISOString() : null,
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
        title: "Started download",
        description:
          "Datasets will show up here once they are fully downloaded",
        status: "info",
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
      initialValues={{
        selectedTickers: [],
        interval: "1h",
        start_date: null,
        end_date: null,
      }}
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

            <div style={{ display: "flex", gap: "8px", marginTop: "16px" }}>
              <div>
                <WithLabel label={"Start date"}>
                  <Field name={"start_date"} component={FormikDatePicker} />
                </WithLabel>
              </div>
              <div>
                <WithLabel label={"End date"}>
                  <Field name={"end_date"} component={FormikDatePicker} />
                </WithLabel>
              </div>
            </div>

            <WithLabel
              label={"Use futures data"}
              containerStyles={{ marginTop: "8px" }}
            >
              <Switch
                isChecked={useFutures}
                onChange={() => setUseFutures(!useFutures)}
              />
            </WithLabel>

            <Button mt={4} type="submit">
              Submit
            </Button>
          </Box>
        </Form>
      )}
    </Formik>
  );
};
const FormStateStocks = ({ modalClose }) => {
  const toast = useToast();
  const nyseSymbolsQuery = useNyseSymbolList();

  if (nyseSymbolsQuery.isLoading) {
    return (
      <Box>
        <Spinner />
      </Box>
    );
  }

  const submitForm = async (values: { selectedTickers: OptionType[] }) => {
    const symbols = values.selectedTickers;
    const promises: Promise<ApiResponse>[] = [];

    symbols.map((item) => {
      const res = saveYfinanceKlines(item.label);
      promises.push(res);
    });

    try {
      await Promise.all(promises);
      toast({
        title: "Started download",
        description:
          "Datasets will show up here once they are fully downloaded",
        status: "info",
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
    <Box>
      <Formik
        initialValues={{ selectedTickers: [] }}
        onSubmit={(values) => submitForm(values)}
      >
        {({ setFieldValue }) => (
          <Form>
            <Box>
              <FormControl>
                <FormLabel fontSize={"x-large"}>
                  Pairs to be downloaded
                </FormLabel>
                <Field
                  name="selectedTickers"
                  as={SelectWithTextFilter}
                  options={nyseSymbolsQuery.data?.map((item) => {
                    return {
                      value: item,
                      label: item,
                    };
                  })}
                  onChange={(selectedOptions: MultiValue<OptionType>) =>
                    setFieldValue("selectedTickers", selectedOptions)
                  }
                  isMulti={true}
                  closeMenuOnSelect={false}
                />
              </FormControl>
              <Button mt={4} type="submit">
                Submit
              </Button>
            </Box>
          </Form>
        )}
      </Formik>
    </Box>
  );
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
          <FormStateStocks modalClose={modalClose} />
        )}
      </div>
    </div>
  );
};

interface CheckedTables {
  tableName: string;
  isChecked: boolean;
}

export const BrowseDatasetsPage = () => {
  const { data, isLoading, refetch } = useDatasetsQuery();
  const { isOpen, jsxContent, setContent, modalClose } = useModal();

  const [checkedTables, setCheckedTables] = useState<CheckedTables[]>([]);

  const forceUpdate = useForceUpdate();
  const toast = useToast();
  const deleteConfirm = useModal();

  const checkBoxOnClick = (tableName: string) => {
    checkedTables.forEach((item) => {
      if (item.tableName === tableName) {
        item.isChecked = !item.isChecked;
      }
    });
    forceUpdate();
  };

  const refetchTables = () => {
    refetch();
  };

  useEffect(() => {
    if (data) {
      const newState = data.map((item) => {
        return {
          tableName: item.table_name,

          isChecked: false,
        };
      });
      setCheckedTables(newState);
      forceUpdate();
    }
  }, [data]);

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

    if (!data) {
      return null;
    }

    return (
      <div>
        <DatasetTable tables={data} checkBoxOnClick={checkBoxOnClick} />
      </div>
    );
  };

  const deleteDataframes = async () => {
    if (!deleteConfirm.isOpen) {
      deleteConfirm.onOpen();
      return;
    }

    const datasets = checkedTables
      .filter((item) => item.isChecked)
      .map((item) => item.tableName);

    const res = await removeDatasets(datasets);

    if (res.status === 200) {
      toast({
        title: `Deleted ${datasets.length} datasets`,
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      refetch();
    }
    deleteConfirm.onClose();
  };

  return (
    <div>
      <ConfirmModal {...deleteConfirm} onConfirm={deleteDataframes} />
      <ChakraModal
        isOpen={isOpen}
        title="New dataset"
        onClose={modalClose}
        modalContentStyle={{ maxWidth: "50%" }}
      >
        {jsxContent}
      </ChakraModal>

      <Box
        display={"flex"}
        alignItems={"center"}
        justifyContent={"space-between"}
      >
        <Button
          onClick={() =>
            setContent(<GetNewDatasetModal modalClose={modalClose} />)
          }
        >
          Add from API
        </Button>

        {checkedTables.filter((item) => item.isChecked).length > 0 && (
          <Button variant={BUTTON_VARIANTS.grey} onClick={deleteDataframes}>
            Delete ({checkedTables.filter((item) => item.isChecked).length})
          </Button>
        )}
      </Box>
      <Box marginTop={"16px"}>{renderDatasetsContainer()}</Box>
    </div>
  );
};

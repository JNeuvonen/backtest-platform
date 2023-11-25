import React, { useState } from "react";
import { useDatasetsQuery } from "../clients/queries";
import { Box, Button, Spinner, Text } from "@chakra-ui/react";
import { DatasetTable } from "../components/tables/Dataset";
import { useModal } from "../hooks/useOpen";
import { ChakraModal } from "../components/chakra/modal";
import { AddIcon } from "@chakra-ui/icons";
import { BasicCard } from "../components/Card";

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

const FormStateBinance = () => {
  return <Box></Box>;
};

const FormStateStocks = () => {
  return <Box></Box>;
};

const STEP_1 = "select-provider";
const STEP_2 = "select-data";
const GetNewDatasetModal = () => {
  const [formState, setFormState] = useState(STEP_1);
  const [dataProvider, setDataProvider] = useState("");

  const advanceStepOne = (selectedProvider: string) => {
    setFormState(STEP_2);
    setDataProvider(selectedProvider);
  };

  return (
    <div style={{ width: "100%" }}>
      <div style={{ margin: "0 auto", width: "max-content" }}>
        {formState === STEP_1 && dataProvider === "Stocks" && (
          <FormStateSelectProvider advanceFormState={advanceStepOne} />
        )}
        {formState === STEP_2 && dataProvider === "Binance" && (
          <FormStateBinance />
        )}
        {formState === STEP_2 && dataProvider === "Stocks" && (
          <FormStateStocks />
        )}
      </div>
    </div>
  );
};

export const AvailablePage = () => {
  const { data, isLoading } = useDatasetsQuery();
  const { isOpen, jsxContent, setContent, modalClose } = useModal(false);

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
        modalContentStyle={{ maxWidth: "50%", paddingBottom: "100px" }}
      >
        {jsxContent}
      </ChakraModal>
      <h1>Available datasets</h1>
      <Button onClick={() => setContent(<GetNewDatasetModal />)}>
        Build new
      </Button>
      {renderDatasetsContainer()}
    </div>
  );
};

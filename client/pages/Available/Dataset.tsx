import React, { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useDatasetQuery } from "../../clients/queries/queries";
import { Box, Button, Spinner, useToast } from "@chakra-ui/react";
import { GenericTable } from "../../components/tables/GenericTable";
import { ChakraModal } from "../../components/chakra/modal";
import { useModal } from "../../hooks/useOpen";
import { BUTTON_VARIANTS } from "../../theme";
import { ColumnModal } from "./ColumnModal";
import { ConfirmInput } from "../../components/form/confirm-input";
import { buildRequest } from "../../clients/fetch";
import { URLS } from "../../clients/endpoints";
import { replaceNthPathItem } from "../../utils/path";
import { useMessageListener } from "../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../../utils/constants";

type DatasetDetailParams = {
  datasetName: string;
};

export const DatasetDetailPage = () => {
  const params = useParams<DatasetDetailParams>();
  const toast = useToast();
  const datasetName = params.datasetName || "";
  const { isOpen, modalClose, setIsOpen } = useModal(false);
  const { data, isLoading, refetch } = useDatasetQuery(datasetName);
  const [selectedColumn, setSelectedColumn] = useState("");
  const navigate = useNavigate();
  const [inputDatasetName, setInputDatasetName] = useState(datasetName);
  const columnOnClickFunction = (selectedColumn: string) => {
    setSelectedColumn(selectedColumn);
    setIsOpen(true);
  };

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_dataset,
    messageCallback: refetch,
  });

  if (isLoading) {
    return <Spinner />;
  }

  const dataset = data?.res.dataset;

  if (!dataset) {
    return <Box>Page is not available</Box>;
  }

  const renameDataset = (newDatasetName: string) => {
    const res = buildRequest({
      url: URLS.set_dataset_name(datasetName),
      method: "PUT",
      payload: { new_dataset_name: newDatasetName },
    });

    res
      .then((res) => {
        if (res?.status === 200) {
          toast({
            title: "Changed the datasets name",
            status: "success",
            duration: 5000,
            isClosable: true,
          });
          setInputDatasetName(newDatasetName);
          navigate(replaceNthPathItem(0, newDatasetName));
        } else {
          toast({
            title: "Failed to change the datasets name",
            status: "error",
            duration: 5000,
            isClosable: true,
          });
        }
      })
      .catch((error) => {
        toast({
          title: "Error",
          description: error?.message,
          status: "error",
          duration: 5000,
          isClosable: true,
        });
      });
  };

  const columns = dataset.columns;

  return (
    <div>
      <ChakraModal
        isOpen={isOpen}
        title={`Column ${selectedColumn}`}
        onClose={modalClose}
        modalContentStyle={{
          minWidth: "max-content",
          maxWidth: "80%",
          marginTop: "10%",
        }}
      >
        <ColumnModal
          columnName={selectedColumn}
          setColumnName={setSelectedColumn}
          datasetName={datasetName}
          close={modalClose}
        />
      </ChakraModal>
      <Box
        display={"flex"}
        justifyContent={"space-between"}
        alignItems={"center"}
      >
        <Box>
          <ConfirmInput
            inputCurrent={inputDatasetName}
            setInputCurrent={setInputDatasetName}
            defaultValue={datasetName}
            newInputCallback={renameDataset}
            message={
              <>
                <span>
                  Are you sure you want to rename dataset to{" "}
                  <b>{inputDatasetName}</b>?
                </span>
              </>
            }
          />
        </Box>
        <Box display={"flex"} gap={"16px"}>
          <Button variant={BUTTON_VARIANTS.grey}>Add dataset</Button>
          <Button variant={BUTTON_VARIANTS.grey}>Add columns</Button>
        </Box>
      </Box>
      <Box marginTop={"16px"}>
        <Box>Tail</Box>
        <GenericTable
          columns={columns}
          rows={dataset.tail}
          columnOnClickFunc={columnOnClickFunction}
        />
      </Box>
    </div>
  );
};

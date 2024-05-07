import { Button, Checkbox, useDisclosure, useToast } from "@chakra-ui/react";
import React, { useState } from "react";
import { BUTTON_VARIANTS } from "../theme";
import { ChakraModal } from "./chakra/modal";
import { CodeEditor } from "./CodeEditor";
import { CODE_PRESET_CATEGORY } from "../utils/constants";
import { ChakraInput } from "./chakra/input";
import { useDataTransformations } from "../clients/queries/queries";
import { createDataTransformation } from "../clients/requests";
import { DataTransformation } from "../clients/queries/response-types";
import { GenericTable } from "./tables/GenericTable";
import { COLOR_BG_PRIMARY_SHADE_ONE } from "../utils/colors";
import { RiDeleteBin2Fill } from "react-icons/ri";
import { TriangleDownIcon, TriangleUpIcon } from "@chakra-ui/icons";

interface Props {
  selectedTransformations: number[];
  onSelect: (transformation: number[]) => void;
}

export interface CreateDataTransformationBody {
  name: string;
  transformation_code: string;
}

const CreateDataTransformationForm = ({
  onSuccessCallback,
}: {
  onSuccessCallback: () => void;
}) => {
  const [createColumnsCode, setCreateColumnsCode] = useState("");
  const [transformationName, setTransformationName] = useState("");
  const toast = useToast();

  const onSubmit = async () => {
    const body: CreateDataTransformationBody = {
      name: transformationName,
      transformation_code: createColumnsCode,
    };

    const res = await createDataTransformation(body);

    if (res.status === 200) {
      toast({
        title: "Created data transformation",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      onSuccessCallback();
    }
  };

  return (
    <div>
      <ChakraInput
        label="Transformation name"
        onChange={setTransformationName}
        containerStyle={{ width: "100%" }}
      />
      <CodeEditor
        code={createColumnsCode}
        setCode={setCreateColumnsCode}
        style={{ marginTop: "16px" }}
        fontSize={13}
        label="Create columns"
        codeContainerStyles={{ width: "100%" }}
        height={"400px"}
        presetCategory={CODE_PRESET_CATEGORY.backtest_create_columns}
      />

      <div
        style={{
          marginTop: "16px",
          display: "flex",
          justifyContent: "flex-end",
        }}
      >
        <Button
          onClick={onSubmit}
          isDisabled={!createColumnsCode || !transformationName}
        >
          Submit
        </Button>
      </div>
    </div>
  );
};

const SelectTransformationsModal = ({
  transformations,
  selectedTransformationIds,
  onSelect,
  onClose,
}: {
  transformations: DataTransformation[];
  selectedTransformationIds: number[];
  onSelect: (newSelectedIds: number[]) => void;
  onClose: () => void;
}) => {
  const [textFilter, setTextFilter] = useState("");
  const [inspectedCode, setInspectedCode] = useState("");
  const inspectCodeModal = useDisclosure();

  const [newSelectedTransformationsState, setNewSelectedTransformationsState] =
    useState(selectedTransformationIds);

  return (
    <div>
      <ChakraModal
        {...inspectCodeModal}
        title="Code"
        modalContentStyle={{ maxWidth: "70%", marginTop: "10%" }}
      >
        <pre>{inspectedCode}</pre>
      </ChakraModal>
      <ChakraInput label="Filter" onChange={setTextFilter} />
      <div style={{ marginTop: "16px", maxHeight: "600px", overflowY: "auto" }}>
        <GenericTable
          columns={["", "Name", "Inspect code"]}
          rows={transformations
            .filter((item) => {
              if (!item.name) return false;
              if (!textFilter) return true;
              return item.name.toLowerCase().includes(textFilter.toLowerCase());
            })
            .map((item) => {
              const found = newSelectedTransformationsState.filter(
                (id) => id === item.id
              );
              return [
                <Checkbox
                  isChecked={found.length > 0}
                  onChange={() => {
                    setNewSelectedTransformationsState((currentSelected) => {
                      if (found.length > 0) {
                        return currentSelected.filter((id) => id !== item.id);
                      } else {
                        return [...currentSelected, item.id];
                      }
                    });
                  }}
                />,
                item.name,
                <Button
                  variant={BUTTON_VARIANTS.nofill}
                  onClick={() => {
                    inspectCodeModal.onOpen();
                    setInspectedCode(item.transformation_code);
                  }}
                >
                  Code
                </Button>,
              ];
            })}
        />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "16px",
        }}
      >
        <Button variant={BUTTON_VARIANTS.nofill}>Cancel</Button>
        <Button
          onClick={() => {
            onSelect(newSelectedTransformationsState);
            onClose();
          }}
        >
          Select
        </Button>
      </div>
    </div>
  );
};

export const DataTransformationControls = (props: Props) => {
  const createTransformationModal = useDisclosure();
  const selectTransformationsModal = useDisclosure();
  const dataTransformationsQuery = useDataTransformations();

  const moveItemUp = (itemIdx: number) => {
    if (itemIdx === 0) {
      return;
    }

    const arrCopy = [...props.selectedTransformations];

    const helper = arrCopy[itemIdx - 1];
    arrCopy[itemIdx - 1] = arrCopy[itemIdx];
    arrCopy[itemIdx] = helper;
    props.onSelect(arrCopy);
  };
  const moveItemDown = (itemIdx: number) => {
    const arrLength = props.selectedTransformations.length;
    if (itemIdx >= arrLength - 1) {
      return;
    }

    const arrCopy = [...props.selectedTransformations];

    const helper = arrCopy[itemIdx + 1];
    arrCopy[itemIdx + 1] = arrCopy[itemIdx];
    arrCopy[itemIdx] = helper;
    props.onSelect(arrCopy);
  };

  const deleteItemFromList = (itemIdx: number) => {
    const arrCopy = [...props.selectedTransformations].filter(
      (item, idx) => idx !== itemIdx
    );
    props.onSelect(arrCopy);
  };

  return (
    <div>
      <ChakraModal
        {...createTransformationModal}
        title="Create transformation"
        modalContentStyle={{ maxWidth: "80%", marginTop: "5%" }}
      >
        <CreateDataTransformationForm
          onSuccessCallback={() => {
            createTransformationModal.onClose();
            dataTransformationsQuery.refetch();
          }}
        />
      </ChakraModal>
      <ChakraModal
        {...selectTransformationsModal}
        title="Select transformations"
        modalContentStyle={{ maxWidth: "80%", marginTop: "5%" }}
      >
        <SelectTransformationsModal
          transformations={
            dataTransformationsQuery.data?.filter((item) => {
              if (!item.name) return false;
              return true;
            }) || []
          }
          selectedTransformationIds={props.selectedTransformations}
          onSelect={props.onSelect}
          onClose={selectTransformationsModal.onClose}
        />
      </ChakraModal>
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={selectTransformationsModal.onOpen}
        >
          Select transformations
        </Button>
        <Button
          variant={BUTTON_VARIANTS.nofill}
          onClick={createTransformationModal.onOpen}
        >
          Create transformation
        </Button>
      </div>

      <div
        style={{
          marginTop: "16px",
          gap: "8px",
          display: "flex",
          flexDirection: "column",
          maxHeight: "500px",
          overflowY: "auto",
        }}
      >
        {props.selectedTransformations.map((item, idx) => {
          const transformationObject = dataTransformationsQuery.data?.filter(
            (transformation) => item === transformation.id
          );
          if (!transformationObject || !transformationObject[0]) return null;
          return (
            <div
              key={item}
              style={{
                padding: "16px",
                background: COLOR_BG_PRIMARY_SHADE_ONE,
                display: "flex",
                justifyContent: "space-between",
                borderRadius: "7px",
              }}
            >
              <div style={{ display: "flex", gap: "16px" }}>
                <div>
                  <TriangleUpIcon
                    style={{ cursor: "pointer" }}
                    onClick={() => moveItemUp(idx)}
                  />
                  <TriangleDownIcon
                    style={{ cursor: "pointer" }}
                    onClick={() => moveItemDown(idx)}
                  />
                </div>
                <div>
                  {idx + 1}. {transformationObject[0].name}
                </div>
              </div>
              <div>
                <RiDeleteBin2Fill
                  style={{ cursor: "pointer" }}
                  onClick={() => deleteItemFromList(idx)}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

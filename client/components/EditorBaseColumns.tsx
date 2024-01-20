import React from "react";
import { getParenthesisSize } from "../utils/content";
import { BUTTON_VARIANTS, TEXT_VARIANTS } from "../theme";
import { useEditorContext } from "../context/editor";
import Title from "./typography/Title";
import { Text } from "@chakra-ui/react";
import { Button, Checkbox, Spinner } from "@chakra-ui/react";
import { AiFillDelete, AiOutlineClose, AiOutlineRight } from "react-icons/ai";
import { ChakraDivider } from "./Divider";
import {
  areAllNestedValuesNull,
  getNonnullEntriesCount,
  isObjectEmpty,
} from "../utils/object";
import { SelectDatasetColumn } from "./SelectDatasetColumn";
import { createScssClassName } from "../utils/css";
import { usePathParams } from "../hooks/usePathParams";
import { useModal } from "../hooks/useOpen";
import { SelectDataset } from "./SelectDataset";
import { useNavigate } from "react-router-dom";
import { MultiValue, SingleValue } from "react-select";
import { OptionType } from "./SelectFilter";
import { ChakraPopover } from "./chakra/popover";
import { ChakraModal } from "./chakra/modal";
import { ColumnInfo } from "./ColumnInfo";
import { getDatasetEditorUrl } from "../utils/navigate";

const CONTAINERS = {
  combine_datasets: "combine-datasets",
  base_dataset: "base",
  all_columns: "all-columns",
};

interface RouteParams {
  datasetName: string;
}

export const EditorBaseColumns = () => {
  const navigate = useNavigate();
  const { datasetName } = usePathParams<RouteParams>();
  const {
    isOpen: selectDatasetOpen,
    modalClose: selectDatasetClose,
    setIsOpen: selectDatasetSetOpen,
  } = useModal();

  const {
    isOpen: columnModalIsOpen,
    modalClose: columnModalOnClose,
    setIsOpen: columnModalSetOpen,
    setSelectedItem: setColumnItem,
    selectedItem: columnItem,
  } = useModal();

  const {
    setIsDelMode,
    dataset,
    isDelMode,
    setDeleteColumns,
    deleteColumns,
    forceUpdate,
    selectedColumns,
    moveColumnsBackToNew,
    selectFromAdded,
    allDatasets,
  } = useEditorContext();

  if (!dataset) {
    return <Spinner />;
  }

  const openColumnModal = (columnName: string) => {
    columnModalSetOpen(true);
    setColumnItem(columnName);
  };

  const removeFromBaseButton = () => {
    if (
      isObjectEmpty(selectedColumns.current) ||
      areAllNestedValuesNull(selectedColumns.current)
    )
      return null;
    return (
      <Button
        rightIcon={<AiOutlineRight />}
        height="32px"
        variant={BUTTON_VARIANTS.grey}
        onClick={moveColumnsBackToNew}
        style={{ marginTop: "8px" }}
      >
        Remove
      </Button>
    );
  };

  return (
    <>
      <ChakraModal
        isOpen={columnModalIsOpen}
        onClose={columnModalOnClose}
        title={columnItem}
      >
        <ColumnInfo datasetName={datasetName} columnName={columnItem} />
      </ChakraModal>
      <div
        className={createScssClassName([
          CONTAINERS.combine_datasets,
          CONTAINERS.base_dataset,
        ])}
      >
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <ChakraPopover
              isOpen={selectDatasetOpen}
              setOpen={() => selectDatasetSetOpen(true)}
              onClose={selectDatasetClose}
              body={
                <SelectDataset
                  onSelect={(
                    selectedItem:
                      | SingleValue<OptionType>
                      | MultiValue<OptionType>
                  ) => {
                    const item = selectedItem as SingleValue<OptionType>;
                    navigate(getDatasetEditorUrl(item?.value as string));
                    selectDatasetClose();
                  }}
                  cancelCallback={selectDatasetClose}
                  datasets={allDatasets.filter(
                    (item) => item.table_name !== datasetName
                  )}
                />
              }
              headerText="Select dataset"
            >
              <Button variant={BUTTON_VARIANTS.nofill}>
                {datasetName} {getParenthesisSize(dataset.columns.length)}
              </Button>
            </ChakraPopover>
            {!isDelMode ? (
              <Button
                variant={BUTTON_VARIANTS.grey}
                height={"30px"}
                padding="10px"
                onClick={() => {
                  setIsDelMode(true);
                  setDeleteColumns([]);
                }}
              >
                <AiFillDelete size={18} />
              </Button>
            ) : (
              <Button
                variant={BUTTON_VARIANTS.grey}
                height={"30px"}
                padding="10px"
                onClick={() => setIsDelMode(false)}
              >
                <AiOutlineClose />
              </Button>
            )}
          </div>
          <ChakraDivider />
        </div>

        <div style={{ marginTop: 8 }}>
          {dataset.columns.map((item) => {
            return (
              <div
                key={item}
                style={{ display: "flex", alignItems: "center", gap: "8px" }}
              >
                {isDelMode && (
                  <Checkbox
                    isChecked={deleteColumns.includes(item)}
                    onChange={() => {
                      if (deleteColumns.includes(item)) {
                        const arr = deleteColumns.filter((col) => col !== item);
                        setDeleteColumns(arr);
                        forceUpdate();
                      } else {
                        const arr = deleteColumns;
                        arr.push(item);
                        setDeleteColumns(arr);
                        forceUpdate();
                      }
                    }}
                  />
                )}

                <Text
                  variant={TEXT_VARIANTS.clickable}
                  onClick={() => openColumnModal(item)}
                >
                  {item}
                </Text>
              </div>
            );
          })}
        </div>

        <div style={{ marginTop: 16 }}>
          <Title
            style={{
              fontWeight: 600,
              fontSize: 17,
            }}
          >
            New columns{" "}
            {getParenthesisSize(
              getNonnullEntriesCount(selectedColumns.current)
            )}
          </Title>
          <ChakraDivider />
        </div>
        {removeFromBaseButton()}

        <div>
          {allDatasets.map((item, i: number) => {
            const columns = selectedColumns.current[item.table_name];

            if (
              !columns ||
              Object.values(columns).every((value) => value === null)
            )
              return null;
            return (
              <SelectDatasetColumn
                defaultOpen={true}
                dataset={item}
                key={i}
                style={i !== 0 ? { marginTop: 16 } : undefined}
                selectedColumns={columns}
                selectColumn={selectFromAdded}
              />
            );
          })}
        </div>
      </div>
    </>
  );
};

import React from "react";
import { getParenthesisSize } from "../utils/content";
import { BUTTON_VARIANTS } from "../theme";
import { useEditorContext } from "../context/editor";
import Title from "./Title";
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
import { PATHS, PATH_KEYS } from "../utils/constants";
import { useModal } from "../hooks/useOpen";
import { ChakraModal } from "./chakra/modal";
import { SelectDataset } from "./SelectDataset";
import { useNavigate } from "react-router-dom";
import { MultiValue, SingleValue } from "react-select";
import { OptionType } from "./SelectFilter";

const CONTAINERS = {
  combine_datasets: "combine-datasets",
  base_dataset: "base",
  all_columns: "all-columns",
};

export const EditorBaseColumns = () => {
  const navigate = useNavigate();
  const datasetName = usePathParams({ key: PATH_KEYS.dataset });
  const { isOpen, modalClose, setIsOpen } = useModal(false);
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
      <ChakraModal isOpen={isOpen} title="Select dataset" onClose={modalClose}>
        <SelectDataset
          onSelect={(
            selectedItem: SingleValue<OptionType> | MultiValue<OptionType>
          ) => {
            const item = selectedItem as SingleValue<OptionType>;
            navigate(
              PATHS.datasets.editor.replace(
                PATH_KEYS.dataset,
                item?.value as string
              )
            );
            modalClose();
          }}
          cancelCallback={modalClose}
          datasets={allDatasets.filter(
            (item) => item.table_name !== datasetName
          )}
        />
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
            <Title
              style={{
                fontWeight: 600,
                fontSize: 17,
              }}
            >
              <Button
                variant={BUTTON_VARIANTS.nofill}
                onClick={() => setIsOpen(true)}
              >
                {datasetName} {getParenthesisSize(dataset.columns.length)}
              </Button>
            </Title>
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
                {item}
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

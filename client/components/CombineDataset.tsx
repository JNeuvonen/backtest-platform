import React, { useEffect, useRef, useState } from "react";
import { useDatasetsQuery } from "../clients/queries/queries";
import { createScssClassName } from "../utils/css";
import { SelectDatasetColumn } from "./SelectDatasetColumn";
import { useForceUpdate } from "../hooks/useForceUpdate";
import { Search } from "./Search";
import cloneDeep from "lodash/cloneDeep";
import {
  AiFillDelete,
  AiOutlineClose,
  AiOutlineLeft,
  AiOutlineRight,
} from "react-icons/ai";
import { Button, Checkbox } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import Title from "./Title";
import { ChakraDivider } from "./Divider";
import {
  areAllNestedValuesNull,
  areAllValuesNull,
  getNonnullEntriesCount,
  isObjectEmpty,
  isOneNestedValueTrue,
} from "../utils/object";
import { useKeyListener } from "../hooks/useKeyListener";
import { useAppContext } from "../context/app";
import { KEYBIND_MSGS, getParenthesisSize } from "../utils/content";
import { ChakraTooltip } from "./Tooltip";
import { ConfirmModal } from "./form/confirm";
import { useModal } from "../hooks/useOpen";
import { addColumnsToDataset } from "../clients/requests";
import { useMessageListener } from "../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS } from "../utils/constants";
import { QueryObserverResult, RefetchOptions } from "@tanstack/react-query";
import { DatasetResponse } from "../clients/queries/response-types";
import { buildRequest } from "../clients/fetch";
import { URLS } from "../clients/endpoints";

interface Props {
  dataset: {
    name: string;
    columns: string[];
    timeseries_col: string;
  };
  refetchParent?: (
    options?: RefetchOptions | undefined
  ) => Promise<QueryObserverResult<DatasetResponse, unknown>>;
}

const CONTAINERS = {
  combine_datasets: "combine-datasets",
  base_dataset: "base",
  all_columns: "all-columns",
};

export type SelectedDatasetColumns = { [key: string]: boolean | null };
export type AddColumnsReqPayload = { table_name: string; columns: string[] }[];
type ColumnsDict = { [key: string]: SelectedDatasetColumns };

export const CombineDataset = ({ dataset, refetchParent }: Props) => {
  const { data, refetch } = useDatasetsQuery();
  const { platform } = useAppContext();
  const {
    isOpen: addModalIsOpen,
    modalClose: addModalClose,
    setIsOpen: addModalSetOpen,
  } = useModal(false);

  const allColumnsData = useRef<ColumnsDict>({});
  const filteredColumns = useRef<ColumnsDict>({});
  const selectedColumns = useRef<ColumnsDict>({});

  const [isDelMode, setIsDelMode] = useState(false);
  const [deleteColumns, setDeleteColumns] = useState<string[]>([]);

  const [componentReady, setComponentReady] = useState(false);
  const forceUpdate = useForceUpdate();

  const handleKeyPress = (event: KeyboardEvent) => {
    if (!isSaveDisabled()) {
      if (platform === "macos") {
        if (event.metaKey && event.key === "s") {
          event.preventDefault();
          addModalSetOpen(true);
        }
      } else {
        if (event.ctrlKey && event.key === "s") {
          event.preventDefault();
          addModalSetOpen(true);
        }
      }
    }
  };

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_all_datasets,
    messageCallback: () => {
      if (refetchParent) refetchParent();
      refetch();
    },
  });
  useKeyListener({ eventAction: handleKeyPress });

  useEffect(() => {
    if (data) {
      allColumnsData.current = {};
      filteredColumns.current = {};
      selectedColumns.current = {};
      data.res.tables.map((item) => {
        if (item.table_name === dataset.name) {
          return;
        }
        allColumnsData.current[item.table_name] = {};
        item.columns.map((col) => {
          if (item.timeseries_col !== col) {
            allColumnsData.current[item.table_name][col] = false;
          }
        });
      });
      filteredColumns.current = cloneDeep(allColumnsData.current);
      setComponentReady(true);
    }
  }, [data, setComponentReady, dataset]);

  if (!data || !componentReady) {
    return <div>No datasets available</div>;
  }

  const selectFromNew = (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => {
    filteredColumns.current[tableName][columnName] = newValue;
    allColumnsData.current[tableName][columnName] = newValue;
    forceUpdate();
  };

  const selectFromAdded = (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => {
    selectedColumns.current[tableName][columnName] = newValue;
    forceUpdate();
  };

  const onSubmit = async () => {
    const reqPayload: AddColumnsReqPayload = [];
    for (const [key, value] of Object.entries(selectedColumns.current)) {
      reqPayload.push({
        table_name: key,
        columns: Object.keys(value),
      });
    }
    const res = await addColumnsToDataset(dataset.name, reqPayload);

    if (res.status === 200) {
      addModalClose();
    }
  };

  const onDatasetSearch = (searchTerm: string) => {
    filteredColumns.current = cloneDeep(allColumnsData.current);
    if (!searchTerm) return forceUpdate();
    Object.keys(filteredColumns.current).forEach((key) => {
      if (!key.includes(searchTerm)) {
        delete filteredColumns.current[key];
      }
    });
    forceUpdate();
  };

  const insertToColumnsStateDict = (
    stateDictRef: React.MutableRefObject<ColumnsDict>,
    tableName: string,
    columnName: string
  ) => {
    if (stateDictRef.current[tableName]) {
      stateDictRef.current[tableName][columnName] = true;
    } else {
      stateDictRef.current[tableName] = {};
      stateDictRef.current[tableName][columnName] = true;
    }
  };

  const moveColumnsToBase = () => {
    for (const [key, value] of Object.entries(allColumnsData.current)) {
      for (const [colName, colValue] of Object.entries(value)) {
        if (colValue) {
          insertToColumnsStateDict(selectedColumns, key, colName);
          allColumnsData.current[key][colName] = null;
          filteredColumns.current[key][colName] = null;
        }
      }
    }
    forceUpdate();
  };

  const moveColumnsBackToNew = () => {
    for (const [key, value] of Object.entries(selectedColumns.current)) {
      for (const [colName, colValue] of Object.entries(value)) {
        if (colValue) {
          insertToColumnsStateDict(allColumnsData, key, colName);
          insertToColumnsStateDict(filteredColumns, key, colName);
          selectedColumns.current[key][colName] = null;
        }
      }
    }
    forceUpdate();
  };

  const allDatasets = data.res.tables;

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

  const isSaveDisabled = () => {
    return (
      isObjectEmpty(selectedColumns.current) ||
      areAllNestedValuesNull(selectedColumns.current)
    );
  };

  const isDeleteDisabled = () => {
    return deleteColumns.length === 0;
  };

  const submitDeleteCols = () => {
    buildRequest({
      url: URLS.delete_dataset_cols(dataset.name),
      payload: {
        cols: deleteColumns,
      },
      method: "POST",
    }).then(() => {
      if (refetchParent) refetchParent();
    });
  };

  return (
    <div>
      <ConfirmModal
        isOpen={addModalIsOpen}
        onClose={addModalClose}
        title="Confirm"
        message={
          <span>
            Are you sure you want to add{" "}
            {getNonnullEntriesCount(selectedColumns.current)} column(s) to the
            dataset?
          </span>
        }
        confirmText="Submit"
        cancelText="Cancel"
        onConfirm={onSubmit}
      />
      <div style={{ display: "flex", gap: "8px" }}>
        <ChakraTooltip label={KEYBIND_MSGS.get_save(platform)}>
          <Button
            style={{ height: "35px", marginBottom: "16px" }}
            isDisabled={isSaveDisabled()}
            onClick={() => addModalSetOpen(true)}
          >
            Save
          </Button>
        </ChakraTooltip>

        {isDelMode && (
          <ChakraTooltip label={KEYBIND_MSGS.get_save(platform)}>
            <Button
              style={{ height: "35px", marginBottom: "16px" }}
              isDisabled={isDeleteDisabled()}
              onClick={submitDeleteCols}
              variant={BUTTON_VARIANTS.grey2}
            >
              Delete {getParenthesisSize(deleteColumns.length)}
            </Button>
          </ChakraTooltip>
        )}
      </div>
      <div className={CONTAINERS.combine_datasets}>
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
                Base columns {getParenthesisSize(dataset.columns.length)}
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
                          const arr = deleteColumns.filter(
                            (col) => col !== item
                          );
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
            {allDatasets.map((item, i) => {
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
        <div
          className={createScssClassName([
            CONTAINERS.combine_datasets,
            CONTAINERS.all_columns,
          ])}
        >
          <div>
            <Button
              leftIcon={<AiOutlineLeft />}
              height="32px"
              variant={BUTTON_VARIANTS.grey}
              onClick={moveColumnsToBase}
              isDisabled={!isOneNestedValueTrue(filteredColumns.current)}
            >
              Add
            </Button>
            <Search
              onSearch={onDatasetSearch}
              placeholder="Search for dataset"
              searchMode="onChange"
              style={{
                width: "300px",
                marginBottom: "8px",
                height: "32px",
                marginTop: "8px",
              }}
            />
          </div>
          <div>
            {allDatasets.map((item, i) => {
              const columns = filteredColumns.current[item.table_name];
              if (!columns || areAllValuesNull(columns)) return null;
              return (
                <SelectDatasetColumn
                  dataset={item}
                  key={i}
                  style={i !== 0 ? { marginTop: 16 } : undefined}
                  selectedColumns={filteredColumns.current[item.table_name]}
                  selectColumn={selectFromNew}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

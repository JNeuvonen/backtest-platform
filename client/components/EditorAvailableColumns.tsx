import React from "react";
import { createScssClassName } from "../utils/css";
import { Button } from "@chakra-ui/react";
import { AiOutlineLeft } from "react-icons/ai";
import { BUTTON_VARIANTS } from "../theme";
import { areAllValuesNull, isOneNestedValueTrue } from "../utils/object";
import { useEditorContext } from "../context/editor";
import { Search } from "./Search";
import { SelectDatasetColumn } from "./SelectDatasetColumn";

const CONTAINERS = {
  combine_datasets: "combine-datasets",
  base_dataset: "base",
  all_columns: "all-columns",
};

export const EditorAvailableColumns = () => {
  const {
    filteredColumns,
    moveColumnsToBase,
    onDatasetSearch,
    allDatasets,
    selectFromNew,
  } = useEditorContext();
  return (
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
  );
};

import React, { useEffect, useRef, useState } from "react";
import { useDatasetsQuery } from "../clients/queries/queries";
import { createScssClassName } from "../utils/css";
import { SelectDatasetColumn } from "./SelectDatasetColumn";
import { useForceUpdate } from "../hooks/useForceUpdate";
import { Search } from "./Search";
import cloneDeep from "lodash/cloneDeep";
import { AiOutlineLeft } from "react-icons/ai";
import { Button } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";

interface Props {
  baseDataset: string;
  baseDatasetColumns: string[];
}

const CONTAINERS = {
  combine_datasets: "combine-datasets",
  base_dataset: "base",
  all_columns: "all-columns",
};

export type SelectedDatasetColumns = { [key: string]: boolean };
type ColumnsDict = { [key: string]: SelectedDatasetColumns };

export const CombineDataset = ({ baseDataset, baseDatasetColumns }: Props) => {
  const { data } = useDatasetsQuery();
  const columnsData = useRef<ColumnsDict>({});
  const selectedColumns = useRef<ColumnsDict>({});

  const [componentReady, setComponentReady] = useState(false);
  const forceUpdate = useForceUpdate();

  useEffect(() => {
    if (data) {
      columnsData.current = {};
      data.res.tables.map((item) => {
        columnsData.current[item.table_name] = {};
        item.columns.map((col) => {
          columnsData.current[item.table_name][col] = false;
        });
      });
      selectedColumns.current = cloneDeep(columnsData.current);
      setComponentReady(true);
    }
  }, [data, setComponentReady]);

  if (!data || !componentReady) {
    return <div>No datasets available</div>;
  }

  const selectColumn = (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => {
    selectedColumns.current[tableName][columnName] = newValue;
    columnsData.current[tableName][columnName] = newValue;
    forceUpdate();
  };

  const onDatasetSearch = (searchTerm: string) => {
    selectedColumns.current = cloneDeep(columnsData.current);
    if (!searchTerm) return forceUpdate();
    Object.keys(selectedColumns.current).forEach((key) => {
      if (!key.includes(searchTerm)) {
        delete selectedColumns.current[key];
      }
    });
    forceUpdate();
  };

  const allDatasets = data.res.tables;
  return (
    <div>
      <div className={CONTAINERS.combine_datasets}>
        <div
          className={createScssClassName([
            CONTAINERS.combine_datasets,
            CONTAINERS.base_dataset,
          ])}
        >
          {baseDatasetColumns.map((item) => {
            return <div key={item}>{item}</div>;
          })}
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
            >
              Add selected
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
              const columns = selectedColumns.current[item.table_name];
              if (!columns) return null;
              return (
                <SelectDatasetColumn
                  dataset={item}
                  key={i}
                  style={i !== 0 ? { marginTop: 16 } : undefined}
                  selectedColumns={selectedColumns.current[item.table_name]}
                  selectColumn={selectColumn}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

import React, { CSSProperties, useState } from "react";
import { DatasetMetadata } from "../clients/queries/response-types";
import { createScssClassName } from "../utils/css";
import { AiFillCaretLeft } from "react-icons/ai";
import { COLOR_CONTENT_PRIMARY } from "../utils/colors";
import { Button, Checkbox } from "@chakra-ui/react";
import { SelectedDatasetColumns } from "./CombineDataset";

const CONTAINERS = {
  dataset_item: "dataset-item",
  header: "header",
};

interface Props {
  dataset: DatasetMetadata;
  style?: CSSProperties;
  selectedColumns: SelectedDatasetColumns;
  defaultOpen?: boolean;
  selectColumn: (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => void;
}

export const SelectDatasetColumn = ({
  dataset,
  style,
  selectedColumns,
  selectColumn,
  defaultOpen = false,
}: Props) => {
  const [expand, setExpand] = useState(defaultOpen);
  return (
    <div className={CONTAINERS.dataset_item} style={style}>
      <div
        className={createScssClassName([
          CONTAINERS.dataset_item,
          CONTAINERS.header,
        ])}
        onClick={() => setExpand(!expand)}
        onKeyDown={(e) => e.key === "Enter" && setExpand(!expand)}
        role="button"
        tabIndex={0}
      >
        {expand ? (
          <AiFillCaretLeft
            fill={COLOR_CONTENT_PRIMARY}
            style={{ transform: "rotate(180deg)" }}
          />
        ) : (
          <AiFillCaretLeft fill={COLOR_CONTENT_PRIMARY} />
        )}
        {dataset.table_name}
      </div>
      {expand && (
        <div>
          {dataset.columns.map((column, index) => {
            const colValue = selectedColumns[column];
            if (colValue === null || colValue === undefined) {
              return null;
            }
            return (
              <div
                key={index}
                style={{
                  marginLeft: "32px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <Checkbox
                  isChecked={colValue}
                  onChange={() =>
                    selectColumn(
                      dataset.table_name,
                      column,
                      !selectedColumns[column]
                    )
                  }
                />
                {column}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

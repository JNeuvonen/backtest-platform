import { Heading } from "@chakra-ui/react";
import React, { useState } from "react";
import { useDatasetsQuery } from "../../clients/queries/queries";
import { GenericTable } from "../../components/tables/GenericTable";
import { Link } from "react-router-dom";
import { PATHS, PATH_KEYS } from "../../utils/constants";
import { ChakraInput } from "../../components/chakra/input";
import useQueryParams from "../../hooks/useQueryParams";
import { DatasetMetadata } from "../../clients/queries/response-types";
import { UI_BACKTEST_MODES } from ".";

const COLUMNS = ["Dataset name"];

interface QueryParams {
  mode: string;
}

export const SimulateSelectDataset = () => {
  const { data } = useDatasetsQuery();

  const { mode } = useQueryParams<QueryParams>();

  const [textFilter, setTextFilter] = useState("");

  const getToLinkBasedOnMode = (item: DatasetMetadata) => {
    if (mode === UI_BACKTEST_MODES.simple) {
      return PATHS.simulate.dataset.replace(PATH_KEYS.dataset, item.table_name);
    }

    if (mode === UI_BACKTEST_MODES.machine_learning) {
      return PATHS.simulate.machine_learning.replace(
        PATH_KEYS.dataset,
        item.table_name
      );
    }
    return "/";
  };

  return (
    <div>
      <div style={{ margin: "0 auto", maxWidth: "1000px", marginTop: "16px" }}>
        <Heading size={"lg"}>Select dataset</Heading>

        <div style={{ marginTop: "8px" }}>
          <ChakraInput
            label={"Filter"}
            onChange={(newVal: string) => setTextFilter(newVal)}
          />
        </div>

        {data && (
          <div style={{ marginTop: "16px" }}>
            <GenericTable
              columns={COLUMNS}
              rows={data
                .filter((item) => {
                  if (!textFilter) return true;
                  return item.table_name
                    .toLowerCase()
                    .includes(textFilter.toLowerCase());
                })
                .map((item) => [
                  <Link
                    to={getToLinkBasedOnMode(item)}
                    className="link-default"
                  >
                    {item.table_name}
                  </Link>,
                ])}
            />
          </div>
        )}
      </div>
    </div>
  );
};

import { Heading } from "@chakra-ui/react";
import React, { useState } from "react";
import { useDatasetsQuery } from "../../clients/queries/queries";
import { GenericTable } from "../../components/tables/GenericTable";
import { Link } from "react-router-dom";
import { PATHS, PATH_KEYS } from "../../utils/constants";
import { ChakraInput } from "../../components/chakra/input";

const COLUMNS = ["Dataset name"];

export const SimulateIndex = () => {
  const { data } = useDatasetsQuery();

  const [textFilter, setTextFilter] = useState("");

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
                    to={PATHS.simulate.dataset.replace(
                      PATH_KEYS.dataset,
                      item.table_name
                    )}
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

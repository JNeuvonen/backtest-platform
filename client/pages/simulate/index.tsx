import { Heading } from "@chakra-ui/react";
import React from "react";
import { useDatasetsQuery } from "../../clients/queries/queries";
import { GenericTable } from "../../components/tables/GenericTable";
import { Link } from "react-router-dom";
import { PATHS, PATH_KEYS } from "../../utils/constants";

const COLUMNS = ["Dataset name"];

export const SimulateIndex = () => {
  const { data } = useDatasetsQuery();

  return (
    <div>
      <div style={{ margin: "0 auto", maxWidth: "1000px", marginTop: "16px" }}>
        <Heading size={"lg"}>Select dataset</Heading>

        {data && (
          <div style={{ marginTop: "16px" }}>
            <GenericTable
              columns={COLUMNS}
              rows={data.map((item) => [
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

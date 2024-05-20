import React, { useState } from "react";
import { useBacktestContext } from ".";
import { ChakraModal } from "../../components/chakra/modal";
import { useDatasetQuery } from "../../clients/queries/queries";
import { OverflopTooltip } from "../../components/OverflowTooltip";
import { Spinner, useDisclosure } from "@chakra-ui/react";
import { usePathParams } from "../../hooks/usePathParams";
import { ColumnInfoModal } from "../../components/ColumnInfoModal";

type PathParams = {
  datasetName: string;
};

export const ShowColumnsModal = () => {
  const { datasetName } = usePathParams<PathParams>();
  const { data } = useDatasetQuery(datasetName);
  const { showColumnsModal: columnsModal } = useBacktestContext();
  const columnDetailsModal = useDisclosure();

  const [selectedColumnName, setSelectedColumnName] = useState("");

  if (!data) return <Spinner />;

  return (
    <ChakraModal {...columnsModal} title="Columns">
      <div id={"COLUMN_MODAL"}>
        {data.columns.map((item, idx: number) => {
          return (
            <div
              key={idx}
              className="link-default"
              onClick={() => {
                setSelectedColumnName(item);
                columnDetailsModal.onOpen();
              }}
            >
              <OverflopTooltip text={item} containerId="COLUMN_MODAL">
                <div>{item}</div>
              </OverflopTooltip>
            </div>
          );
        })}
      </div>
      {selectedColumnName && (
        <ChakraModal
          {...columnDetailsModal}
          title={`Column ${selectedColumnName}`}
          modalContentStyle={{ maxWidth: "80%", marginTop: "5%" }}
        >
          <ColumnInfoModal
            datasetName={datasetName}
            columnName={selectedColumnName}
          />
        </ChakraModal>
      )}
    </ChakraModal>
  );
};

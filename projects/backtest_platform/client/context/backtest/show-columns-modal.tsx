import { UseDisclosureReturn, useDisclosure } from "@chakra-ui/react";
import React, { useState } from "react";
import { ChakraModal } from "../../components/chakra/modal";
import { OverflopTooltip } from "../../components/OverflowTooltip";
import { ColumnInfoModal } from "../../components/ColumnInfoModal";

interface Props {
  datasetName: string;
  columns: string[];
  columnsModal: UseDisclosureReturn;
}

export const ShowColumnModal = ({
  columns: data,
  datasetName,
  columnsModal,
}: Props) => {
  const [selectedColumnName, setSelectedColumnName] = useState("");
  const columnDetailsModal = useDisclosure();
  return (
    <ChakraModal {...columnsModal} title="Columns">
      <div id={"COLUMN_MODAL"}>
        {data.map((item, idx) => {
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

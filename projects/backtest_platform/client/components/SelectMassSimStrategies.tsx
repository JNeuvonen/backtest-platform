import { Button, Checkbox, Spinner, useDisclosure } from "@chakra-ui/react";
import React from "react";
import { useRuleBasedMassBacktests } from "../clients/queries/queries";
import { ChakraModal } from "./chakra/modal";
import { BUTTON_VARIANTS } from "../theme";
import { GenericTable } from "./tables/GenericTable";

interface Props {
  onSelect: (selectedIds: number[]) => void;
  selectedIds: number[];
}

const TABLE_COLUMNS = ["", "Strategy name", "Result"];

export const SelectMassSimStrategies = ({ onSelect, selectedIds }: Props) => {
  const modalControls = useDisclosure();
  const backtestsQuery = useRuleBasedMassBacktests();

  if (!backtestsQuery.data) {
    return (
      <div>
        <ChakraModal
          {...modalControls}
          title="Code"
          modalContentStyle={{ maxWidth: "70%", marginTop: "10%" }}
        >
          <Spinner />
        </ChakraModal>
      </div>
    );
  }

  return (
    <div>
      <ChakraModal
        {...modalControls}
        title="Code"
        modalContentStyle={{ maxWidth: "70%", marginTop: "10%" }}
      >
        <div>
          <GenericTable
            columns={TABLE_COLUMNS}
            rows={backtestsQuery.data
              .filter((item) => {
                if (!item.name) {
                  return false;
                }
                return true;
              })
              .map((item) => {
                return [
                  <Checkbox
                    isChecked={selectedIds.includes(item.id)}
                    onChange={() => {
                      const isIncluded = selectedIds.includes(item.id);
                      let newSelectedIds: number[];
                      if (isIncluded) {
                        newSelectedIds = selectedIds.filter(
                          (id) => id !== item.id
                        );
                      } else {
                        newSelectedIds = [...selectedIds, item.id];
                      }
                      onSelect(newSelectedIds);
                    }}
                  />,
                  item.name,
                  item.end_balance,
                ];
              })}
          />
        </div>
      </ChakraModal>
      <Button variant={BUTTON_VARIANTS.nofill} onClick={modalControls.onOpen}>
        Select strategies
      </Button>

      {backtestsQuery.data.map((item, idx) => {
        if (!selectedIds.includes(item.id)) return null;
        return (
          <div>
            {idx + 1}. {item.name}
          </div>
        );
      })}
    </div>
  );
};

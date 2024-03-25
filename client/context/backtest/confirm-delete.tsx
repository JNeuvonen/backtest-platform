import React from "react";
import { useBacktestContext } from ".";
import { ConfirmModal } from "../../components/form/Confirm";
import { deleteManyBacktests } from "../../clients/requests";
import { useToast } from "@chakra-ui/react";

export const ConfirmDeleteSelectedModal = () => {
  const {
    confirmDeleteSelectedModal,
    selectedBacktests,
    datasetBacktestsQuery,
    resetSelection,
    onDeleteMode,
  } = useBacktestContext();
  const toast = useToast();
  return (
    <div>
      <ConfirmModal
        {...confirmDeleteSelectedModal}
        onConfirm={async () => {
          const res = await deleteManyBacktests(selectedBacktests);
          if (res.status === 200) {
            datasetBacktestsQuery.refetch();
            confirmDeleteSelectedModal.onClose();
            toast({
              title: "Deleted backtests",
              status: "info",
              duration: 5000,
              isClosable: true,
            });
            onDeleteMode.onClose();
            resetSelection();
          }
        }}
        title={"Confirm action"}
        message={`Are you sure you want to delete ${selectedBacktests.length} backtests?`}
        confirmText="Delete"
      />
    </div>
  );
};

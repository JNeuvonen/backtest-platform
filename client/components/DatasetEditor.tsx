import React from "react";
import { Spinner } from "@chakra-ui/react";
import { getNonnullEntriesCount } from "../utils/object";
import { ConfirmModal } from "./form/Confirm";
import { useEditorContext } from "../context/editor";
import { EditorToolbar } from "./EditorToolbar";
import { EditorBaseColumns } from "./EditorBaseColumns";
import { EditorAvailableColumns } from "./EditorAvailableColumns";

const CONTAINERS = {
  combine_datasets: "combine-datasets",
  base_dataset: "base",
  all_columns: "all-columns",
};

export type SelectedDatasetColumns = { [key: string]: boolean | null };

export const DatasetEditor = () => {
  const {
    selectedColumns,
    deleteColumns,
    addModalIsOpen,
    addModalClose,
    delModalIsOpen,
    delModalClose,
    onSubmit,
    providerMounted,
    data,
    dataset,
    submitDeleteCols,
  } = useEditorContext();

  if (!data || !providerMounted) {
    return <div>No datasets available</div>;
  }

  if (!dataset) {
    return <Spinner />;
  }

  return (
    <div>
      <ConfirmModal
        isOpen={addModalIsOpen}
        onClose={addModalClose}
        title="Confirm"
        message={
          <span>
            Are you sure you want to add{" "}
            {getNonnullEntriesCount(selectedColumns.current)} column(s) to the
            dataset?
          </span>
        }
        confirmText="Submit"
        cancelText="Cancel"
        onConfirm={onSubmit}
      />

      <ConfirmModal
        isOpen={delModalIsOpen}
        onClose={delModalClose}
        title="Warning"
        message={
          <span>
            Are you sure you want to delete {deleteColumns.length} column(s)
            from the dataset?
          </span>
        }
        confirmText="Delete"
        cancelText="Cancel"
        onConfirm={submitDeleteCols}
      />
      <EditorToolbar />
      <div className={CONTAINERS.combine_datasets}>
        <EditorBaseColumns />
        <EditorAvailableColumns />
      </div>
    </div>
  );
};

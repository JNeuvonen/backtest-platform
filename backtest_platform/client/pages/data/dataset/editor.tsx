import React from "react";
import { EditorProvider } from "../../../context/editor";
import { DatasetEditor } from "../../../components/DatasetEditor";

export const DatasetEditorPage = () => {
  return (
    <EditorProvider>
      <DatasetEditor />
    </EditorProvider>
  );
};

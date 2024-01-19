import React from "react";
import { OnMount } from "@monaco-editor/react";
import { PythonEditor } from "./PythonEditor";

interface Props {
  code: string;
  setCode: React.Dispatch<React.SetStateAction<string>>;
}

export const RunPythonOnAllCols = ({ code, setCode }: Props) => {
  const handleCodeChange = (newValue: string | undefined) => {
    setCode(newValue ?? "");
  };
  const handleEditorDidMount: OnMount = (editor) => {
    editor.setValue(code);
    editor.setPosition({ lineNumber: 10, column: 20 });
    editor.focus();
  };

  return (
    <div>
      <div
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          display: "flex",
        }}
      >
        <PythonEditor
          code={code}
          onChange={handleCodeChange}
          editorMount={handleEditorDidMount}
          height={"400px"}
          containerStyles={{ width: "65%", height: "100%" }}
          fontSize={13}
        />

        <div>Code presets will come here</div>
      </div>
    </div>
  );
};

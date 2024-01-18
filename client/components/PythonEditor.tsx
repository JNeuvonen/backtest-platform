import React, { CSSProperties } from "react";
import { Editor, EditorProps, OnMount } from "@monaco-editor/react";

interface Props {
  code: string;
  height?: string;
  onChange: (value: string | undefined) => void;
  editorMount: OnMount;
  containerStyles?: CSSProperties;
  fontSize?: number;
}

export const PythonEditor = ({
  code,
  onChange,
  editorMount,
  height = "400px",
  containerStyles = {},
  fontSize = 20,
}: Props) => {
  const editorOptions: EditorProps["options"] = {
    minimap: { enabled: false },
    fontSize: fontSize,
  };
  return (
    <div style={containerStyles}>
      <Editor
        height={height}
        defaultLanguage="python"
        theme="vs-dark"
        value={code}
        onChange={onChange}
        onMount={editorMount}
        width={"100%"}
        options={editorOptions}
      />
    </div>
  );
};

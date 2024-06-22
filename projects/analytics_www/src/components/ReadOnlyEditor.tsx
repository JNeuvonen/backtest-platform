import { Editor } from "@monaco-editor/react";
import { CSSProperties } from "react";

interface Props {
  value: string;
  height?: number;
  style?: CSSProperties;
}

export const ReadOnlyEditor = ({ value, height = 400, style }: Props) => {
  return (
    <div style={style}>
      <Editor
        height={height}
        defaultLanguage="python"
        theme="vs-dark"
        value={value}
        onChange={() => {}}
        onMount={() => {}}
        width={"100%"}
        options={{ minimap: { enabled: false }, readOnly: true }}
      />
    </div>
  );
};

import React from "react";
import { OnMount } from "@monaco-editor/react";
import { PythonEditor } from "./PythonEditor";
import { ChakraSelect } from "./chakra/select";
import { DOM_IDS, NULL_FILL_STRATEGIES } from "../utils/constants";
import { CodeEditor } from "./CodeInput";

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
      <ChakraSelect
        label={"Null fill strategy"}
        options={NULL_FILL_STRATEGIES}
        id={DOM_IDS.select_null_fill_strat}
        defaultValueIndex={3}
      />
      <CodeEditor code={code} setCode={setCode} />
    </div>
  );
};

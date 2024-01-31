import React from "react";
import { ChakraSelect } from "./chakra/Select";
import { DOM_IDS, NULL_FILL_STRATEGIES } from "../utils/constants";
import { CodeEditor } from "./CodeEditor";

interface Props {
  code: string;
  setCode: React.Dispatch<React.SetStateAction<string>>;
}

export const RunPythonOnAllCols = ({ code, setCode }: Props) => {
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

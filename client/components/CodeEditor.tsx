import { Editor, EditorProps, OnMount } from "@monaco-editor/react";
import React, { CSSProperties } from "react";
import { CodePresets } from "./CodePresets";
import { FormControl, FormLabel } from "@chakra-ui/react";
import { SaveCodePreset } from "./SaveCodePresetPopover";
import { SelectCodePreset } from "./CodePresetPopover";

interface Props {
  code: string;
  setCode?: React.Dispatch<React.SetStateAction<string>>;
  style?: CSSProperties;
  codeContainerStyles?: CSSProperties;
  presets?: string;
  height?: string;
  fontSize?: number;
  editorDidMount?: OnMount;
  label?: JSX.Element | string;
  readOnly?: boolean;
  disableCodePresets?: boolean;
  autoFocus?: boolean;
  usePresets?: boolean;
  presetCategory?: string;
}

export const CodeEditor = ({
  code,
  setCode,
  style,
  codeContainerStyles = { width: "65%", height: "100%" },
  height = "400px",
  fontSize = 20,
  editorDidMount,
  label,
  readOnly = false,
  disableCodePresets = false,
  autoFocus = true,
  usePresets = true,
  presetCategory,
}: Props) => {
  const handleCodeChange = (newValue: string | undefined) => {
    if (setCode) {
      setCode(newValue ?? "");
    }
  };

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    if (editorDidMount) {
      editorDidMount(editor, monaco);
    } else {
      editor.setValue(code);
      editor.setPosition({ lineNumber: 100, column: 200 });
      if (autoFocus) {
        editor.focus();
      }
    }
  };

  const editorOptions: EditorProps["options"] = {
    minimap: { enabled: false },
    fontSize: fontSize,
    readOnly,
  };

  if (label) {
    return (
      <div
        style={{
          ...style,
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <FormControl style={codeContainerStyles}>
          <FormLabel>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <div>{label}</div>
              {usePresets && (
                <>
                  <SelectCodePreset
                    presetCategory={presetCategory || ""}
                    onPresetSelect={(selectedPreset: string) => {
                      if (setCode) {
                        setCode(selectedPreset);
                      }
                    }}
                  />
                  <SaveCodePreset
                    presetCategory={presetCategory || ""}
                    code={code}
                    onSaveCodePreset={(selectedPreset: string) => {
                      if (setCode) {
                        setCode(selectedPreset);
                      }
                    }}
                  />
                </>
              )}
            </div>
          </FormLabel>
          <Editor
            height={height}
            defaultLanguage="python"
            theme="vs-dark"
            value={code}
            onChange={handleCodeChange}
            onMount={handleEditorDidMount}
            width={"100%"}
            options={editorOptions}
          />
        </FormControl>

        {!disableCodePresets && <CodePresets />}
      </div>
    );
  }
  return (
    <div
      style={{
        ...style,
        display: "flex",
        alignItems: "center",
        gap: "8px",
      }}
    >
      <div style={codeContainerStyles}>
        <Editor
          height={height}
          defaultLanguage="python"
          theme="vs-dark"
          value={code}
          onChange={handleCodeChange}
          onMount={handleEditorDidMount}
          width={"100%"}
          options={editorOptions}
        />
      </div>

      {!disableCodePresets && <CodePresets />}
    </div>
  );
};

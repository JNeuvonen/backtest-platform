import { FormControl, FormLabel, Input } from "@chakra-ui/react";
import React, { CSSProperties, useState } from "react";
import { ChakraCheckbox } from "../chakra/checkbox";

export interface CheckboxValue {
  label: string;
  isChecked: boolean;
}

interface Props {
  options: CheckboxValue[];
  onSelect: (item: CheckboxValue) => void;
  searchFilterPlaceholder?: string;
  searchFilterStyles?: CSSProperties;
  style?: CSSProperties;
}

export const CheckboxMulti = ({
  options,
  onSelect,
  searchFilterPlaceholder = "Filter by searching",
  searchFilterStyles = { width: "200px" },
  style = {},
}: Props) => {
  const [searchTerm, setSearchTerm] = useState("");

  return (
    <div style={style}>
      <Input
        style={searchFilterStyles}
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder={searchFilterPlaceholder}
      />
      <div
        style={{
          marginTop: "8px",
          display: "flex",
          flexDirection: "column",
          gap: "2px",
          fontSize: "10px",
        }}
      >
        {options.map((item, i) => {
          if (searchTerm && !item.label.includes(searchTerm)) return null;
          return (
            <ChakraCheckbox
              key={i}
              {...item}
              onChange={(newCheckedValue) => {
                item["isChecked"] = newCheckedValue;
                onSelect(item);
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

import React, { CSSProperties, KeyboardEvent, useState } from "react";
import { Input, FormControl } from "@chakra-ui/react";

interface SearchComponentProps {
  onSearch: (searchTerm: string) => void;
  searchMode: "onEnter" | "onChange";
  placeholder?: string;
  style?: CSSProperties;
  className?: string;
}

export const Search: React.FC<SearchComponentProps> = ({
  onSearch,
  placeholder = "Search...",
  style,
  className,
  searchMode,
}) => {
  const [value, setValue] = useState("");
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      onSearch(value);
    }
  };

  const onChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    setValue(newValue);
    if (searchMode === "onEnter") return;
    onSearch(newValue);
  };

  return (
    <FormControl
      as="form"
      onSubmit={(e) => {
        e.preventDefault();
        onSearch(value);
      }}
    >
      <Input
        value={value}
        placeholder={placeholder}
        onChange={onChange}
        style={style}
        className={className}
        onKeyDown={handleKeyDown}
        size="md"
      />
    </FormControl>
  );
};

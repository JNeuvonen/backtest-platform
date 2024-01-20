import React, { CSSProperties } from "react";
interface Props {
  children: React.ReactNode[] | React.ReactNode;
  style?: CSSProperties;
}
export const ToolBarStyle = ({ children, style }: Props) => {
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: "8px", ...style }}
    >
      {children}
    </div>
  );
};

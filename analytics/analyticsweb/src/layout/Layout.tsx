import React from "react";
import { SideNav } from ".";

interface Props {
  children: React.ReactNode;
}

export const LayoutContainer = ({ children }: Props) => {
  return (
    <div className="layout">
      <SideNav />
      <div className="layout__content">{children}</div>
    </div>
  );
};

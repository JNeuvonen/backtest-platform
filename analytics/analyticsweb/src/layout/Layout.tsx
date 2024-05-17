import React from "react";
import { SideNav } from ".";

interface Props {
  children: React.ReactNode;
}

export const LayoutContainer = ({ children }: Props) => {
  return (
    <div>
      <SideNav />
      <div>{children}</div>
    </div>
  );
};

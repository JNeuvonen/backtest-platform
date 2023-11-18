import React from "react";
import { SideNav } from "./SideNav";

export const Layout = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="layout">
      <SideNav />
      <div className="layout__content">{children}</div>
    </div>
  );
};

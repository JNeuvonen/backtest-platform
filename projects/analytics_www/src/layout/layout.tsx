import React from "react";
import { useAppContext } from "src/context";
import { SIDENAV_DEFAULT_WIDTH } from "src/utils";
import { SideNav } from ".";
import { BottomInfoFooter } from "./bottom-footer";

interface Props {
  children: React.ReactNode;
}

export const LayoutContainer = ({ children }: Props) => {
  const { isMobileLayout } = useAppContext();
  return (
    <div className="layout">
      <SideNav />
      <div
        className="layout__content"
        style={{
          marginLeft: isMobileLayout ? 0 : SIDENAV_DEFAULT_WIDTH,
          paddingBottom: "35px",
        }}
      >
        {children}
      </div>
      <BottomInfoFooter />
    </div>
  );
};

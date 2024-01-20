import React, { ReactNode } from "react";

interface Props {
  style?: React.CSSProperties;
  tagType?: "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
  children?: ReactNode;
}

const Title: React.FC<Props> = ({
  style = { fontWeight: 700, fontSize: 28 },
  tagType = "h1",
  children,
}) => {
  const Tag = tagType as keyof JSX.IntrinsicElements;
  return <Tag style={style}>{children}</Tag>;
};

export default Title;

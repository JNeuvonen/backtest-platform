import React from "react";
import { Link } from "react-router-dom";
import { FaExternalLinkAlt } from "react-icons/fa";
import { COLOR_LINK_DEFAULT } from "../utils/colors";
import { open } from "@tauri-apps/api/shell";

const ExternalLink = ({
  to,
  linkText,
  iconSize = "13px",
  iconColor = COLOR_LINK_DEFAULT,
  isExternal = false,
}) => {
  const handleOpenLink = async (url: string) => {
    await open(url);
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <Link
        to={to}
        className="link-default"
        onClick={(e) => {
          if (isExternal) {
            e.preventDefault();
            handleOpenLink(to);
          }
        }}
      >
        {linkText}
      </Link>
      <div>
        <FaExternalLinkAlt color={iconColor} size={iconSize} />
      </div>
    </div>
  );
};

export default ExternalLink;

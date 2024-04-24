import React from "react";
import { Link } from "react-router-dom";
import { FaExternalLinkAlt } from "react-icons/fa";
import { COLOR_LINK_DEFAULT } from "../utils/colors";

const ExternalLink = ({
  to,
  linkText,
  iconSize = "13px",
  iconColor = COLOR_LINK_DEFAULT,
}) => {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <Link to={to} className="link-default">
        {linkText}
      </Link>
      <div>
        <FaExternalLinkAlt color={iconColor} size={iconSize} />
      </div>
    </div>
  );
};

export default ExternalLink;

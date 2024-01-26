import React from "react";
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink } from "@chakra-ui/react";
import { Link } from "react-router-dom";

interface BreadcrumbItem {
  label: string;
  href: string;
}

interface GenericBreadcrumbsProps {
  items: BreadcrumbItem[];
  separator?: string;
}

export const Breadcrumbs: React.FC<GenericBreadcrumbsProps> = ({
  items,
  separator = "/",
}) => (
  <Breadcrumb separator={separator}>
    {items.map((item, index) => (
      <BreadcrumbItem key={index} isCurrentPage={index === items.length - 1}>
        <Link
          to={item.href}
          className="link-default"
          style={{
            color: index === items.length - 1 ? "white" : "#097bed",
            cursor: index === items.length - 1 ? "default" : "pointer",
          }}
        >
          {item.label}
        </Link>
      </BreadcrumbItem>
    ))}
  </Breadcrumb>
);

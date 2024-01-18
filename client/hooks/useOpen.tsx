import React, { useState } from "react";

export const useModal = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [jsxContent, setJsxContent] = useState<null | JSX.Element>(<div></div>);
  const [selectedItem, setSelectedItem] = useState("");

  const setContent = (htmlElement: JSX.Element) => {
    setJsxContent(htmlElement);
    setIsOpen(true);
  };

  const modalClose = () => {
    setJsxContent(null);
    setIsOpen(false);
  };

  return {
    isOpen,
    setIsOpen,
    jsxContent,
    setContent,
    modalClose,
    selectedItem,
    setSelectedItem,
    modalOpen: () => setIsOpen(true),
  };
};

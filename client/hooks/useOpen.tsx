import React, { useState } from "react";

export const useModal = (defaultState: boolean) => {
  const [isOpen, setIsOpen] = useState(defaultState);
  const [jsxContent, setJsxContent] = useState<null | JSX.Element>(<div></div>);

  const setContent = (htmlElement: JSX.Element) => {
    setJsxContent(htmlElement);
    setIsOpen(true);
  };

  const modalClose = () => {
    setJsxContent(null);
    setIsOpen(false);
  };

  return { isOpen, setIsOpen, jsxContent, setContent, modalClose };
};

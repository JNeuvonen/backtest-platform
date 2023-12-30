import { Input } from "@chakra-ui/react";
import React from "react";
import { ConfirmModal } from "./confirm";
import { useModal } from "../../hooks/useOpen";

interface Props {
  defaultValue: string;
  confirmTitle?: string;
  confirmText?: string;
  cancelText?: string;
  message?: React.ReactNode | string;
  newInputCallback: (newInput: string) => void;
  inputCurrent: string;
  setInputCurrent: React.Dispatch<React.SetStateAction<string>>;
}

export const ConfirmInput = ({
  defaultValue,
  confirmTitle = "Confirm",
  confirmText = "Confirm",
  cancelText = "Cancel",
  message = "Are you sure you want to confirm this action?",
  newInputCallback,
  inputCurrent,
  setInputCurrent,
}: Props) => {
  const { isOpen, modalClose, setIsOpen } = useModal(false);
  const handleToggle = () => {
    if (inputCurrent !== defaultValue) {
      setIsOpen(true);
    }
  };
  const handleClose = () => setIsOpen(false);
  const handleConfirm = () => {
    newInputCallback(inputCurrent);
    handleClose();
  };
  return (
    <>
      <Input
        onBlur={handleToggle}
        onChange={(e) => setInputCurrent(e.target.value)}
        value={inputCurrent}
        onKeyDown={(e) => {
          if (e.key === "Enter" && e.target instanceof HTMLInputElement) {
            e.preventDefault();
            e.target.blur();
          }
        }}
      />
      <ConfirmModal
        isOpen={isOpen}
        onClose={modalClose}
        title={confirmTitle}
        confirmText={confirmText}
        cancelText={cancelText}
        message={message}
        onConfirm={handleConfirm}
        cancelCallback={() => {
          setInputCurrent(defaultValue);
          modalClose();
        }}
      />
    </>
  );
};

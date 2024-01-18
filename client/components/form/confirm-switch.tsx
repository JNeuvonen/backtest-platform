import { Switch } from "@chakra-ui/react";
import React from "react";
import { ConfirmModal } from "./confirm";
import { useModal } from "../../hooks/useOpen";

interface Props {
  isChecked: boolean;
  confirmTitle?: string;
  confirmText?: string;
  cancelText?: string;
  message?: string | React.ReactNode;
  setNewToggleValue: (newValue: boolean) => void;
}

export const ConfirmSwitch = ({
  isChecked,
  setNewToggleValue,
  confirmTitle = "Confirm",
  confirmText = "Confirm",
  cancelText = "Cancel",
  message = "Are you sure you want to confirm this action?",
}: Props) => {
  const { isOpen, modalClose, setIsOpen } = useModal();
  const handleToggle = () => setIsOpen(true);
  const handleClose = () => setIsOpen(false);
  const handleConfirm = () => {
    setNewToggleValue(!isChecked);
    handleClose();
  };
  return (
    <>
      <Switch isChecked={isChecked} onChange={handleToggle} />
      <ConfirmModal
        isOpen={isOpen}
        onClose={modalClose}
        title={confirmTitle}
        confirmText={confirmText}
        cancelText={cancelText}
        message={message}
        onConfirm={handleConfirm}
      />
    </>
  );
};

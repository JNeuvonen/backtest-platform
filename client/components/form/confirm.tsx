import React from "react";
import { Button } from "@chakra-ui/react";
import { ChakraModal } from "../chakra/modal";
import { BUTTON_VARIANTS } from "../../theme";

interface ConfirmProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title?: string;
  message?: string | React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  cancelCallback?: () => void;
}

export const ConfirmModal: React.FC<ConfirmProps> = ({
  isOpen,
  onClose,
  title = "Confirm",
  message = "Are you sure you want to confirm this action?",
  confirmText = "Confirm",
  cancelText = "Cancel",
  onConfirm,
  cancelCallback,
}) => {
  return (
    <ChakraModal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      modalContentStyle={{
        marginTop: "25%",
      }}
      footerContent={
        <>
          <Button
            variant={BUTTON_VARIANTS.grey2}
            mr={3}
            onClick={cancelCallback ? cancelCallback : onClose}
          >
            {cancelText}
          </Button>
          <Button colorScheme="blue" onClick={onConfirm}>
            {confirmText}
          </Button>
        </>
      }
    >
      {message}
    </ChakraModal>
  );
};

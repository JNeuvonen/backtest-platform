import React, { ReactNode } from "react";
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  ModalProps,
  ModalContentProps,
} from "@chakra-ui/react";
import { COLOR_BG_PRIMARY } from "src/theme";

interface CustomModalProps extends ModalProps {
  title: string;
  children: ReactNode;
  footerContent?: ReactNode;
  modalContentStyle?: ModalContentProps;
}

export const ChakraModal: React.FC<CustomModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  footerContent,
  modalContentStyle = {},
  ...props
}) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} {...props}>
      <ModalOverlay />
      <ModalContent bg={COLOR_BG_PRIMARY} padding="8px" {...modalContentStyle}>
        <ModalHeader>{title}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>{children}</ModalBody>
        {footerContent && <ModalFooter>{footerContent}</ModalFooter>}
      </ModalContent>
    </Modal>
  );
};

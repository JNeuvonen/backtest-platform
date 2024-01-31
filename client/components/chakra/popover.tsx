import React from "react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverHeader,
  PopoverArrow,
  PopoverCloseButton,
  PopoverBody,
  PopoverFooter,
  PlacementWithLogical,
} from "@chakra-ui/react";
import { COLOR_BG_TERTIARY } from "../../utils/colors";

interface Props {
  children: React.ReactNode;
  body: React.ReactNode;
  footer?: React.ReactNode;
  headerText: string;
  placement?: PlacementWithLogical;
  closeOnBlur?: boolean;
  isOpen: boolean;
  onClose: () => void;
  setOpen: () => void;
}

export const ChakraPopover = ({
  children,
  body,
  footer,
  headerText,
  placement = "bottom",
  closeOnBlur = true,
  isOpen,
  onClose,
  setOpen,
}: Props) => {
  return (
    <Popover
      placement={placement}
      closeOnBlur={closeOnBlur}
      isOpen={isOpen}
      onClose={onClose}
    >
      <PopoverTrigger>
        <span
          onClick={setOpen}
          onKeyDown={(e) => e.key === "Enter" && setOpen()} // Add keyboard listener
          tabIndex={0}
          role="button"
        >
          {children}
        </span>
      </PopoverTrigger>
      <PopoverContent bg={COLOR_BG_TERTIARY} borderColor={COLOR_BG_TERTIARY}>
        <PopoverHeader pt={4} fontWeight="bold" border="0">
          {headerText}
        </PopoverHeader>
        <PopoverArrow bg={COLOR_BG_TERTIARY} />
        <PopoverCloseButton />
        <PopoverBody>{isOpen && body}</PopoverBody>
        {footer && <PopoverFooter>{footer}</PopoverFooter>}
      </PopoverContent>
    </Popover>
  );
};

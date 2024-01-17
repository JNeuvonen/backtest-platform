import React from "react";
import {
  PlacementWithLogical,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverCloseButton,
  PopoverContent,
  PopoverFooter,
  PopoverHeader,
} from "@chakra-ui/react";

interface Props {
  popoverTrigger: React.ReactNode;
  body: React.ReactNode;
  footer?: React.ReactNode;
  headerText: string;
  placement: PlacementWithLogical;
  closeOnBlur?: boolean;
}

export const ChakraPopover = ({
  popoverTrigger,
  body,
  footer,
  headerText,
  placement = "bottom",
  closeOnBlur = true,
}: Props) => {
  return (
    <Popover placement={placement} closeOnBlur={closeOnBlur}>
      {popoverTrigger}
      <PopoverContent>
        <PopoverHeader pt={4} fontWeight="bold" border="0">
          {headerText}
        </PopoverHeader>
        <PopoverArrow />
        <PopoverCloseButton />
        <PopoverBody>{body}</PopoverBody>
        {footer && <PopoverFooter>{footer}</PopoverFooter>}
      </PopoverContent>
    </Popover>
  );
};

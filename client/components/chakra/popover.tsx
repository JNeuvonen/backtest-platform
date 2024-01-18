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
  PopoverTrigger,
} from "@chakra-ui/react";
import { COLOR_BG_TERTIARY } from "../../utils/colors";

interface Props {
  children: React.ReactNode;
  body: React.ReactNode;
  footer?: React.ReactNode;
  headerText: string;
  placement?: PlacementWithLogical;
  closeOnBlur?: boolean;
}

export const ChakraPopover = ({
  children,
  body,
  footer,
  headerText,
  placement = "bottom",
  closeOnBlur = true,
}: Props) => {
  return (
    <Popover placement={placement} closeOnBlur={closeOnBlur}>
      <PopoverTrigger>{children}</PopoverTrigger>
      <PopoverContent bg={COLOR_BG_TERTIARY} borderColor={COLOR_BG_TERTIARY}>
        <PopoverHeader pt={4} fontWeight="bold" border="0">
          {headerText}
        </PopoverHeader>
        <PopoverArrow bg={COLOR_BG_TERTIARY} />
        <PopoverCloseButton />
        <PopoverBody>{body}</PopoverBody>
        {footer && <PopoverFooter>{footer}</PopoverFooter>}
      </PopoverContent>
    </Popover>
  );
};

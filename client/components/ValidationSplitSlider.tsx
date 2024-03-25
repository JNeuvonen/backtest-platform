import React, { CSSProperties } from "react";
import {
  RangeSlider,
  RangeSliderTrack,
  RangeSliderFilledTrack,
  RangeSliderThumb,
  FormControl,
  FormLabel,
} from "@chakra-ui/react";

interface Props {
  sliderValue: number[];
  setSliderValue?:
    | React.Dispatch<React.SetStateAction<number[]>>
    | ((newState: number[]) => void);
  containerStyle?: CSSProperties;
  isReadOnly?: boolean;
  formLabelText?: string;
}

export function ValidationSplitSlider({
  sliderValue,
  setSliderValue,
  containerStyle,
  isReadOnly = false,
  formLabelText = "Validation split",
}: Props) {
  return (
    <FormControl style={containerStyle}>
      <FormLabel>
        {formLabelText}: [{sliderValue[0]}-{sliderValue[1]}]
      </FormLabel>
      <RangeSlider
        min={0}
        max={100}
        value={sliderValue}
        onChange={(val) => {
          if (setSliderValue) setSliderValue(val);
        }}
        isDisabled={isReadOnly}
      >
        <RangeSliderTrack>
          <RangeSliderFilledTrack />
        </RangeSliderTrack>
        <RangeSliderThumb index={0} />
        <RangeSliderThumb index={1} />
      </RangeSlider>
    </FormControl>
  );
}

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
  setSliderValue: React.Dispatch<React.SetStateAction<number[]>>;
  containerStyle?: CSSProperties;
}

export function ValidationSplitSlider({
  sliderValue,
  setSliderValue,
  containerStyle,
}: Props) {
  return (
    <FormControl style={containerStyle}>
      <FormLabel>
        Validation split: [{sliderValue[0]}-{sliderValue[1]}]
      </FormLabel>
      <RangeSlider
        min={0}
        max={100}
        value={sliderValue}
        onChange={(val) => setSliderValue(val)}
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

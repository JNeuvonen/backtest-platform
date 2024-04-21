import React, { CSSProperties } from "react";
import {
  RangeSlider,
  RangeSliderTrack,
  RangeSliderFilledTrack,
  RangeSliderThumb,
  FormControl,
  FormLabel,
} from "@chakra-ui/react";

interface GenericRangeSliderProps {
  minValue: number;
  maxValue: number;
  values: number[];
  onChange: (values: number[]) => void;
  isDisabled?: boolean;
  label?: string;
  containerStyle?: CSSProperties;
}

export const GenericRangeSlider: React.FC<GenericRangeSliderProps> = ({
  minValue,
  maxValue,
  values,
  onChange,
  isDisabled = false,
  label = "Range Slider",
  containerStyle,
}) => {
  return (
    <FormControl style={containerStyle}>
      <FormLabel>
        {label}: [{values[0]}-{values[1]}]
      </FormLabel>
      <RangeSlider
        min={minValue}
        max={maxValue}
        value={values}
        onChange={(val) => onChange(val)}
        isDisabled={isDisabled}
      >
        <RangeSliderTrack>
          <RangeSliderFilledTrack />
        </RangeSliderTrack>
        <RangeSliderThumb index={0} />
        <RangeSliderThumb index={1} />
      </RangeSlider>
    </FormControl>
  );
};

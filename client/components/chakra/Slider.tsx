import React, { CSSProperties } from "react";
import {
  FormControl,
  FormLabel,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  SliderMark,
} from "@chakra-ui/react";

interface ChakraSliderProps {
  label: string;
  id?: string;
  min?: number;
  max?: number;
  step?: number;
  onChange?: (value: number) => void;
  defaultValue?: number;
  containerStyles?: CSSProperties;
  value?: number;
}

export const ChakraSlider: React.FC<ChakraSliderProps> = ({
  label,
  id,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  defaultValue = 50,
  containerStyles,
}) => {
  return (
    <FormControl style={containerStyles}>
      <FormLabel htmlFor={id}>{label}</FormLabel>
      <Slider
        id={id}
        defaultValue={defaultValue}
        min={min}
        max={max}
        step={step}
        onChange={(val) => onChange?.(val)}
      >
        <SliderTrack>
          <SliderFilledTrack />
        </SliderTrack>
        <SliderThumb />
        <SliderMark value={min}>{min}</SliderMark>
        <SliderMark value={max}>{max}</SliderMark>
      </Slider>
    </FormControl>
  );
};

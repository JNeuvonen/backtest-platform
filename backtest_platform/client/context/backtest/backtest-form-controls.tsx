import React from "react";
import { TEXT_VARIANTS } from "../../theme";
import { Text, UseDisclosureReturn } from "@chakra-ui/react";
import { DiskManager } from "../../utils/disk";
import { FormikProps } from "formik";

interface Props {
  backtestDiskManager: DiskManager;
  formikRef: React.RefObject<FormikProps<any>>;
  columnsModal: UseDisclosureReturn;
  forceUpdate: () => void;
}

export const BacktestFormControls = ({
  columnsModal,
  formikRef,
  backtestDiskManager,
  forceUpdate,
}: Props) => {
  return (
    <div style={{ display: "flex", gap: "16px" }}>
      <Text variant={TEXT_VARIANTS.clickable} onClick={columnsModal.onOpen}>
        Show columns
      </Text>
      <Text
        variant={TEXT_VARIANTS.clickable}
        onClick={() => {
          backtestDiskManager.reset();
          formikRef.current?.resetForm();
          forceUpdate();
        }}
      >
        Reset form
      </Text>
    </div>
  );
};

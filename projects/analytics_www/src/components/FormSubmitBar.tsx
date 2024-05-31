import { Button } from "@chakra-ui/react";
import { CSSProperties } from "react";
import { BUTTON_VARIANTS } from "src/theme";

interface Props {
  cancelText?: string;
  submitText?: string;
  submitDisabled?: boolean;
  hideCancel?: boolean;
  cancelCallback?: () => void;
  submitCallback?: () => void;
  style?: CSSProperties;
}

export const FormSubmitBar = ({
  cancelText = "Cancel",
  submitDisabled = false,
  cancelCallback,
  submitText = "Submit",
  submitCallback,
  style = {},
}: Props) => {
  return (
    <div className="form-submit-bar" style={style}>
      <Button variant={BUTTON_VARIANTS.nofill} onClick={cancelCallback}>
        {cancelText}
      </Button>
      <Button
        type="submit"
        onClick={submitCallback}
        isDisabled={submitDisabled}
      >
        {submitText}
      </Button>
    </div>
  );
};

import React from "react";
import DatePicker from "react-datepicker";

export const FormikDatePicker = ({ field, form, ...props }) => {
  return (
    <DatePicker
      {...field}
      {...props}
      selected={(field.value && new Date(field.value)) || null}
      onChange={(date) => form.setFieldValue(field.name, date)}
    />
  );
};

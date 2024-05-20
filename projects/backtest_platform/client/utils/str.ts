export const createPythonCode = (lines: string[]) => {
  return lines.join("\n");
};

export const parseJson = (str: string) => {
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
};

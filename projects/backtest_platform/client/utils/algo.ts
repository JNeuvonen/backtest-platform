interface Dict {
  [key: string]: any;
}

export function binarySearch<T extends Dict>(
  array: T[],
  target: number,
  selector: (item: T) => number
): T | null {
  let low = 0;
  let high = array.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const compare = selector(array[mid]);

    if (compare === target) {
      return array[mid];
    } else if (compare < target) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return null;
}

export const LINE_CHART_COLORS = [
  "#FFD54F",
  "#039BE5",
  "#3949AB",
  "#E57373",
  "#F06292",
  "#BA68C8",
  "#9575CD",
  "#7986CB",
  "#64B5F6",
  "#4FC3F7",
  "#4DD0E1",
  "#4DB6AC",
  "#81C784",
  "#AED581",
  "#DCE775",
  "#FFF176",
  "#FFB74D",
  "#FF8A65",
  "#A1887F",
  "#E0E0E0",
  "#90A4AE",
  "#E53935",
  "#D81B60",
  "#8E24AA",
  "#5E35B1",
  "#1E88E5",
  "#00ACC1",
  "#00897B",
  "#43A047",
  "#7CB342",
  "#C0CA33",
  "#FDD835",
  "#FFB300",
  "#FB8C00",
  "#F4511E",
  "#6D4C41",
  "#757575",
  "#546E7A",
  "#D32F2F",
  "#C2185B",
  "#7B1FA2",
  "#512DA8",
  "#303F9F",
  "#1976D2",
  "#0288D1",
  "#0097A7",
  "#00796B",
  "#388E3C",
  "#689F38",
  "#AFB42B",
  "#FBC02D",
  "#FFA000",
  "#F57C00",
  "#E64A19",
  "#5D4037",
  "#616161",
  "#C62828",
  "#AD1457",
  "#6A1B9A",
  "#4527A0",
  "#283593",
  "#1565C0",
  "#0277BD",
  "#00838F",
  "#00695C",
  "#2E7D32",
  "#558B2F",
  "#9E9D24",
  "#F9A825",
  "#FF8F00",
  "#EF6C00",
  "#D84315",
  "#4E342E",
  "#B71C1C",
  "#880E4F",
  "#4A148C",
  "#311B92",
  "#1A237E",
  "#0D47A1",
  "#01579B",
  "#006064",
  "#004D40",
  "#1B5E20",
  "#33691E",
  "#827717",
  "#F57F17",
  "#FF6F00",
  "#E65100",
  "#BF360C",
  "#FF1744",
  "#F50057",
  "#D500F9",
  "#651FFF",
  "#3D5AFE",
  "#2979FF",
  "#00B0FF",
  "#00E5FF",
  "#1DE9B6",
  "#00E676",
  "#76FF03",
  "#C6FF00",
  "#FFEA00",
  "#FFC400",
  "#FF9100",
  "#FF3D00",
];

export const MAX_NUMBER_OF_LINES = LINE_CHART_COLORS.length - 1;

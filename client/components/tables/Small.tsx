import {
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@chakra-ui/react";

import React, { CSSProperties } from "react";

export type RowItem = string[] | number[] | (string | number[]);

interface Props {
  columns: string[];
  columnOnClickFunc?: (item: string) => void;
  rowOnClickFunc?: (item: RowItem) => void;
  rows: RowItem[];
  containerStyles?: CSSProperties;
}

export const SmallTable = ({
  columns,
  columnOnClickFunc,
  rowOnClickFunc,
  rows,
  containerStyles,
}: Props) => {
  return (
    <TableContainer style={containerStyles}>
      <Table size="sm" className="basic-table">
        <Thead>
          <Tr>
            {columns.map((item) => (
              <Th
                key={item}
                onClick={() => columnOnClickFunc?.(item)}
                style={{
                  cursor: columnOnClickFunc ? "pointer" : undefined,
                }}
              >
                {item}
              </Th>
            ))}
          </Tr>
        </Thead>
        <Tbody>
          {rows.map((row, i: number) => (
            <Tr
              key={i}
              className="table-row-hover" // Add class for hover style
              onClick={() => rowOnClickFunc?.(row)}
              style={{
                cursor: rowOnClickFunc ? "pointer" : undefined,
              }}
            >
              {row.map((rowItem, j: number) => (
                <Td key={`${i}-${j}`}>{rowItem}</Td>
              ))}
            </Tr>
          ))}
        </Tbody>
      </Table>
    </TableContainer>
  );
};

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

interface Props {
  columns: string[];
  columnOnClickFunc?: (item: string) => void;
  rows: string[][] | number[][] | (string | number)[][];
  containerStyles: CSSProperties;
}

export const SmallTable = ({
  columns,
  columnOnClickFunc,
  rows,
  containerStyles,
}: Props) => {
  return (
    <TableContainer style={containerStyles}>
      <Table size="sm" className="basic-table">
        <Thead>
          <Tr>
            {columns.map((item) => {
              return (
                <Th
                  key={item}
                  className={"table-th-hover"}
                  onClick={() => {
                    if (!columnOnClickFunc) return;
                    columnOnClickFunc(item);
                  }}
                >
                  {item}
                </Th>
              );
            })}
          </Tr>
        </Thead>
        <Tbody>
          {rows.map(
            (item: string[] | number[] | (string | number)[], i: number) => {
              return (
                <Tr key={i}>
                  {item.map((rowItem: string | number, j: number) => {
                    return <Td key={`${i}-${j}`}>{rowItem}</Td>;
                  })}
                </Tr>
              );
            }
          )}
        </Tbody>
      </Table>
    </TableContainer>
  );
};

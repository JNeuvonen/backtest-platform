import React from "react";
import { Box, Button, Table, Tbody, Td, Th, Thead, Tr } from "@chakra-ui/react";
import { DatasetMetadata } from "../../clients/queries";

interface Props {
  tables: DatasetMetadata[];
}

const COLUMNS = ["Name", "Start date", "End date", "Columns", ""];

export const DatasetTable = ({ tables }: Props) => {
  return (
    <Box overflow="auto">
      <Table variant="simple">
        <Thead>
          <Tr>
            {COLUMNS.map((item) => {
              return <Th key={item}>{item}</Th>;
            })}
          </Tr>
        </Thead>
        <Tbody>
          {tables.map((item) => {
            return (
              <Tr key={item.table_name}>
                <Td>{item.table_name}</Td>
                <Td>{new Date(item.start_date).toLocaleDateString()}</Td>
                <Td>{new Date(item.end_date).toLocaleDateString()}</Td>
                <Td>
                  <Button variant="grey">View</Button>
                </Td>
                <Td></Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </Box>
  );
};

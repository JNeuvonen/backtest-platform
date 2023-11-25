import React from "react";
import { Box, Button, Table, Tbody, Td, Th, Thead, Tr } from "@chakra-ui/react";
import { DatasetMetadata } from "../../clients/queries";
import { ChakraModal, Modal } from "../chakra/modal";
import { useModal } from "../../hooks/useOpen";

interface Props {
  tables: DatasetMetadata[];
}

const COLUMNS = ["Name", "Start date", "End date", "Columns", ""];

const ColumnsModalContent = ({ columns }: { columns: string[] }) => {
  return (
    <div>
      {columns.map((item) => {
        return <div key={item}>{item}</div>;
      })}
    </div>
  );
};

export const DatasetTable = ({ tables }: Props) => {
  const { isOpen, setContent, modalClose, jsxContent } = useModal(false);
  return (
    <Box overflow="auto">
      <ChakraModal isOpen={isOpen} onClose={modalClose} title={"Table columns"}>
        {jsxContent}
      </ChakraModal>
      <Table variant="simple" className="basic-table">
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
                  <Button
                    variant="grey"
                    onClick={() => {
                      setContent(
                        <ColumnsModalContent columns={item.columns} />
                      );
                    }}
                  >
                    View
                  </Button>
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

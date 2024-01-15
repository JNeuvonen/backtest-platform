import { Box, Button, Table, Tbody, Td, Th, Thead, Tr } from "@chakra-ui/react";
import React from "react";
import { useModal } from "../../hooks/useOpen";
import { ChakraModal } from "../chakra/modal";
import { Link } from "react-router-dom";
import { DatasetMetadata } from "../../clients/queries/response-types";
import { PATHS, PATH_KEYS } from "../../utils/constants";

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
                <Td>
                  <Link
                    to={`${PATHS.datasets.info.replace(
                      PATH_KEYS.dataset,
                      item.table_name
                    )}`}
                    className="link-default"
                  >
                    {item.table_name}
                  </Link>
                </Td>
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

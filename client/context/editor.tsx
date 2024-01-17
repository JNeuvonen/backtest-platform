import React, {
  ReactNode,
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { SelectedDatasetColumns } from "../components/DatasetEditor";
import { useModal } from "../hooks/useOpen";
import { useForceUpdate } from "../hooks/useForceUpdate";
import { addColumnsToDataset } from "../clients/requests";
import { DOM_EVENT_CHANNELS, PATH_KEYS } from "../utils/constants";
import { usePathParams } from "../hooks/usePathParams";
import { areAllNestedValuesNull, isObjectEmpty } from "../utils/object";
import { useAppContext } from "./app";
import { useKeyListener } from "../hooks/useKeyListener";
import { useMessageListener } from "../hooks/useMessageListener";
import { buildRequest } from "../clients/fetch";
import { URLS } from "../clients/endpoints";
import { useToast } from "@chakra-ui/react";
import { useDatasetQuery, useDatasetsQuery } from "../clients/queries/queries";
import cloneDeep from "lodash/cloneDeep";
import {
  Dataset,
  DatasetMetadata,
  DatasetsResponse,
} from "../clients/queries/response-types";

interface EditorContextType {
  filteredColumns: React.MutableRefObject<ColumnsDict>;
  selectedColumns: React.MutableRefObject<ColumnsDict>;
  isDelMode: boolean;
  setIsDelMode: React.Dispatch<React.SetStateAction<boolean>>;
  deleteIsLoading: boolean;
  deleteColumns: string[];
  setDeleteColumns: React.Dispatch<React.SetStateAction<string[]>>;
  addModalIsOpen: boolean;
  addModalClose: () => void;
  addModalSetOpen: React.Dispatch<React.SetStateAction<boolean>>;
  delModalIsOpen: boolean;
  delModalClose: () => void;
  delModalSetOpen: React.Dispatch<React.SetStateAction<boolean>>;
  forceUpdate: () => void;
  selectFromNew: (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => void;
  selectFromAdded: (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => void;
  onSubmit: () => Promise<void>;
  isSaveDisabled: () => boolean;
  insertToColumnsStateDict: (
    stateDictRef: React.MutableRefObject<ColumnsDict>,
    tableName: string,
    columnName: string
  ) => void;
  moveColumnsToBase: () => void;
  moveColumnsBackToNew: () => void;
  isDeleteDisabled: () => boolean;
  dataset: Dataset | null;
  allDatasets: DatasetMetadata[];
  providerMounted: boolean;
  data: DatasetsResponse | undefined;
  submitDeleteCols: () => void;
  onDatasetSearch: (searchTerm: string) => void;
}

export const EditorContext = createContext<EditorContextType>(
  {} as EditorContextType
);

interface Props {
  children: ReactNode;
}

export type ColumnsDict = { [key: string]: SelectedDatasetColumns };

export type AddColumnsReqPayload = { table_name: string; columns: string[] }[];

export const EditorProvider: React.FC<Props> = ({ children }) => {
  const datasetName = usePathParams({ key: PATH_KEYS.dataset });
  const { data: datasetResp, refetch: refetchDataset } =
    useDatasetQuery(datasetName);

  const dataset = datasetResp?.res.dataset ? datasetResp.res.dataset : null;
  const { data, refetch } = useDatasetsQuery();
  const allDatasets = data?.res.tables || [];

  const { platform } = useAppContext();
  const allColumnsData = useRef<ColumnsDict>({});
  const filteredColumns = useRef<ColumnsDict>({});
  const selectedColumns = useRef<ColumnsDict>({});
  const [isDelMode, setIsDelMode] = useState(false);
  const [deleteIsLoading, setDeleteIsLoading] = useState(false);
  const [deleteColumns, setDeleteColumns] = useState<string[]>([]);

  const [providerMounted, setProviderMounted] = useState(false);
  const toast = useToast();

  const {
    isOpen: addModalIsOpen,
    modalClose: addModalClose,
    setIsOpen: addModalSetOpen,
  } = useModal(false);
  const {
    isOpen: delModalIsOpen,
    modalClose: delModalClose,
    setIsOpen: delModalSetOpen,
  } = useModal(false);

  const forceUpdate = useForceUpdate();

  const handleKeyPress = (event: KeyboardEvent) => {
    if (!isSaveDisabled()) {
      if (platform === "macos") {
        if (event.metaKey && event.key === "s") {
          event.preventDefault();
          addModalSetOpen(true);
        }
      } else {
        if (event.ctrlKey && event.key === "s") {
          event.preventDefault();
          addModalSetOpen(true);
        }
      }
    }
  };

  useEffect(() => {
    if (data && datasetResp) {
      allColumnsData.current = {};
      filteredColumns.current = {};
      selectedColumns.current = {};
      data.res.tables.map((item) => {
        if (item.table_name === datasetName) {
          return;
        }
        allColumnsData.current[item.table_name] = {};
        item.columns.map((col) => {
          if (item.timeseries_col !== col) {
            allColumnsData.current[item.table_name][col] = false;
          }
        });
      });
      filteredColumns.current = cloneDeep(allColumnsData.current);
      setProviderMounted(true);
    }
  }, [data, setProviderMounted, datasetResp, datasetName]);

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_all_datasets,
    messageCallback: () => {
      if (refetchDataset) refetchDataset();
      selectedColumns.current = {};
      refetch();
    },
  });

  useKeyListener({ eventAction: handleKeyPress });

  const isSaveDisabled = () => {
    return (
      isObjectEmpty(selectedColumns.current) ||
      areAllNestedValuesNull(selectedColumns.current)
    );
  };

  const selectFromNew = (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => {
    filteredColumns.current[tableName][columnName] = newValue;
    allColumnsData.current[tableName][columnName] = newValue;
    forceUpdate();
  };

  const selectFromAdded = (
    tableName: string,
    columnName: string,
    newValue: boolean
  ) => {
    selectedColumns.current[tableName][columnName] = newValue;
    forceUpdate();
  };

  const onSubmit = async () => {
    const reqPayload: AddColumnsReqPayload = [];
    for (const [key, value] of Object.entries(selectedColumns.current)) {
      reqPayload.push({
        table_name: key,
        columns: Object.keys(value),
      });
    }
    const res = await addColumnsToDataset(datasetName, reqPayload);

    if (res.status === 200) {
      addModalClose();
      forceUpdate();
    }
  };

  const insertToColumnsStateDict = (
    stateDictRef: React.MutableRefObject<ColumnsDict>,
    tableName: string,
    columnName: string
  ) => {
    if (stateDictRef.current[tableName]) {
      stateDictRef.current[tableName][columnName] = true;
    } else {
      stateDictRef.current[tableName] = {};
      stateDictRef.current[tableName][columnName] = true;
    }
  };

  const moveColumnsToBase = () => {
    for (const [key, value] of Object.entries(allColumnsData.current)) {
      for (const [colName, colValue] of Object.entries(value)) {
        if (colValue) {
          insertToColumnsStateDict(selectedColumns, key, colName);
          allColumnsData.current[key][colName] = null;
          filteredColumns.current[key][colName] = null;
        }
      }
    }
    forceUpdate();
  };

  const moveColumnsBackToNew = () => {
    for (const [key, value] of Object.entries(selectedColumns.current)) {
      for (const [colName, colValue] of Object.entries(value)) {
        if (colValue) {
          insertToColumnsStateDict(allColumnsData, key, colName);
          insertToColumnsStateDict(filteredColumns, key, colName);
          selectedColumns.current[key][colName] = null;
        }
      }
    }
    forceUpdate();
  };

  const submitDeleteCols = () => {
    setDeleteIsLoading(true);
    delModalSetOpen(false);
    setIsDelMode(false);

    buildRequest({
      url: URLS.delete_dataset_cols(datasetName),
      payload: {
        cols: deleteColumns,
      },
      method: "POST",
    })
      .then((res) => {
        toast({
          title: `Deleted ${deleteColumns.length} column(s)`,
          status: "info",
          duration: 5000,
          isClosable: true,
        });
        if (res.status === 200) {
          setDeleteColumns([]);
          if (refetchDataset) refetchDataset();
        }

        setDeleteIsLoading(false);
      })
      .catch((error) => {
        toast({
          title: "Error",
          description: error?.message,
          status: "error",
          duration: 5000,
          isClosable: true,
        });
        setDeleteIsLoading(false);
      });
  };

  const onDatasetSearch = (searchTerm: string) => {
    filteredColumns.current = cloneDeep(allColumnsData.current);
    if (!searchTerm) return forceUpdate();
    Object.keys(filteredColumns.current).forEach((key) => {
      if (!key.includes(searchTerm)) {
        delete filteredColumns.current[key];
      }
    });
    forceUpdate();
  };

  const isDeleteDisabled = () => {
    return deleteColumns.length === 0;
  };

  return (
    <EditorContext.Provider
      value={{
        onDatasetSearch,
        providerMounted,
        forceUpdate,
        filteredColumns,
        selectedColumns,
        isDelMode,
        setIsDelMode,
        deleteIsLoading,
        deleteColumns,
        setDeleteColumns,
        addModalIsOpen,
        addModalClose,
        addModalSetOpen,
        delModalIsOpen,
        delModalClose,
        delModalSetOpen,
        selectFromNew,
        selectFromAdded,
        onSubmit,
        isSaveDisabled,
        insertToColumnsStateDict,
        moveColumnsToBase,
        moveColumnsBackToNew,
        isDeleteDisabled,
        dataset,
        allDatasets,
        data,
        submitDeleteCols,
      }}
    >
      {children}
    </EditorContext.Provider>
  );
};

export const useEditorContext = () => useContext(EditorContext);

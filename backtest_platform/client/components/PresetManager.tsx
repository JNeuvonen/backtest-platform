import React, { useMemo, useState } from "react";
import { useCodePresets } from "../clients/queries/queries";
import {
  Button,
  Heading,
  Spinner,
  Text,
  Textarea,
  Tooltip,
  useDisclosure,
  useToast,
} from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { IoMdSettings } from "react-icons/io";
import { ChakraModal } from "./chakra/modal";
import { CodePreset } from "../clients/queries/response-types";
import { ChakraCard } from "./chakra/Card";
import {
  COLOR_BG_SECONDARY,
  COLOR_BG_TERTIARY,
  COLOR_LINK_DEFAULT,
} from "../utils/colors";
import { ChakraInput } from "./chakra/input";
import { CodeEditor } from "./CodeEditor";
import { CODE_PRESET_CATEGORY } from "../utils/constants";
import { FormSubmitBar } from "./form/FormSubmitBar";
import { WithLabel } from "./form/WithLabel";
import { IoTrashBinSharp } from "react-icons/io5";
import { putCodePreset } from "../clients/requests";

interface Props {
  onPresetSelect: (selectedPreset: CodePreset) => void;
  presetCategory: string;
}

const groupByLabels = (presets: CodePreset[]) => {
  const ret: { [key: string]: CodePreset[] } = {};

  presets.forEach((item) => {
    if (item.label && ret[item.label]) {
      ret[item.label].push(item);
    } else if (!item.label) {
      if (ret["No category"]) {
        ret["No category"].push(item);
      } else {
        ret["No category"] = [item];
      }
    } else {
      ret[item.label] = [item];
    }
  });

  return ret;
};

const PresetItem = ({
  preset,
  onSelect,
  refetchCallback,
}: {
  preset: CodePreset;
  onSelect: (preset: CodePreset) => void;
  refetchCallback: () => void;
}) => {
  const [isHovering, setIsHovering] = useState(false);
  const editModal = useDisclosure();
  const [editedCode, setEditedCode] = useState(preset.code);
  const [editedName, setEditedName] = useState(preset.name);
  const [editedDescrp, setEditedDescrp] = useState(preset.description || "");

  const toast = useToast();

  const onEditSubmit = async () => {
    const body = {
      ...preset,
      code: editedCode,
      name: editedName,
      description: editedDescrp,
    };

    const res = await putCodePreset(body);

    if (res.status === 200) {
      toast({
        title: "Edited code preset",
        status: "info",
        duration: 5000,
        isClosable: true,
      });
      editModal.onClose();
    }
  };
  return (
    <>
      <ChakraModal
        {...editModal}
        modalContentStyle={{ maxWidth: "50%", marginTop: "7%" }}
        title={"Edit code preset"}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <ChakraInput
            label={"Name"}
            value={editedName}
            onChange={(newValue) => setEditedName(newValue)}
          />

          <Button
            colorScheme={"red"}
            variant={BUTTON_VARIANTS.dangerFill}
            leftIcon={<IoTrashBinSharp />}
          >
            Delete
          </Button>
        </div>
        <div style={{ marginTop: "16px" }}>
          <WithLabel label={"Description"}>
            <Textarea
              value={editedDescrp}
              onChange={(e) => setEditedDescrp(e.target.value)}
              placeholder="Here is a sample placeholder"
              size="sm"
            />
          </WithLabel>
        </div>
        <div>
          <CodeEditor
            code={editedCode}
            setCode={setEditedCode}
            style={{ marginTop: "16px" }}
            fontSize={13}
            label={`Edit ${preset.name}`}
            codeContainerStyles={{ width: "100%" }}
            height={"65vh"}
            presetCategory={CODE_PRESET_CATEGORY.backtest_create_columns}
            usePresets={false}
          />

          <FormSubmitBar
            style={{ marginTop: "16px" }}
            submitText={"Save"}
            cancelCallback={editModal.onClose}
            submitCallback={onEditSubmit}
          />
        </div>
      </ChakraModal>
      <Tooltip label={preset.description || ""}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: isHovering ? COLOR_BG_TERTIARY : undefined,
            borderRadius: "5px",
            padding: "5px",
          }}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
        >
          <div>
            <Text
              fontSize="sm"
              style={{ width: "max-content", cursor: "pointer" }}
              color={COLOR_LINK_DEFAULT}
              onClick={() => onSelect(preset)}
            >
              {preset.name}
            </Text>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            {isHovering && (
              <>
                <div>
                  <Text
                    fontSize="sm"
                    style={{ width: "max-content", cursor: "pointer" }}
                    color={COLOR_LINK_DEFAULT}
                    onClick={editModal.onOpen}
                  >
                    Edit
                  </Text>
                </div>
              </>
            )}
          </div>
        </div>
      </Tooltip>
    </>
  );
};

const ManageCodePresetsModal = ({
  presets,
  onSelect,
  refetchCallback,
}: {
  presets: CodePreset[];
  onSelect: (preset: CodePreset) => void;
  refetchCallback: () => void;
}) => {
  const [searchTerm, setSearchTerm] = useState("");

  const groups = useMemo(() => {
    const filteredPresets = presets.filter((preset) =>
      preset.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
    return groupByLabels(filteredPresets);
  }, [JSON.stringify(presets), searchTerm]);

  const renderByGroups = () => (
    <div>
      {Object.keys(groups).map((label) => (
        <ChakraCard
          key={label}
          heading={
            <Heading size="md">
              {label.toUpperCase()} ({groups[label].length})
            </Heading>
          }
          containerStyles={{ marginTop: "16px" }}
        >
          <div style={{ marginTop: "16px" }}>
            {groups[label].map((preset, index) => (
              <PresetItem
                key={index}
                preset={preset}
                onSelect={onSelect}
                refetchCallback={refetchCallback}
              />
            ))}
          </div>
        </ChakraCard>
      ))}
    </div>
  );

  return (
    <div>
      <ChakraInput
        label={`Filter ${presets.length} indicators`}
        onChange={(value) => setSearchTerm(value)}
      />
      {renderByGroups()}
    </div>
  );
};

export const ManagePresets = ({ onPresetSelect, presetCategory }: Props) => {
  const codePresetsQuery = useCodePresets();
  const managePresetsModal = useDisclosure();

  if (!codePresetsQuery.data) {
    return <Spinner />;
  }

  const presets = codePresetsQuery.data.filter(
    (item) => item.category === presetCategory
  );

  return (
    <div>
      <Tooltip label={"Manage code presets"}>
        <Button
          variant={BUTTON_VARIANTS.grey2}
          padding={"0px"}
          height={"30px"}
          onClick={managePresetsModal.onOpen}
        >
          <IoMdSettings />
        </Button>
      </Tooltip>
      <ChakraModal
        {...managePresetsModal}
        title={"Code presets"}
        modalContentStyle={{
          maxWidth: "40%",
          marginTop: "5%",
          background: COLOR_BG_SECONDARY,
        }}
      >
        <ManageCodePresetsModal
          presets={presets}
          onSelect={onPresetSelect}
          refetchCallback={() => {
            codePresetsQuery.refetch();
          }}
        />
      </ChakraModal>
    </div>
  );
};

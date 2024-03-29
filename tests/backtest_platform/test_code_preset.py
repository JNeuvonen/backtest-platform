import pytest
from tests.backtest_platform.t_constants import CodePresetId

from tests.backtest_platform.t_utils import Fetch


@pytest.mark.acceptance
def test_setup_sanity(cleanup_db, fixt_create_code_preset):
    preset_id = fixt_create_code_preset
    code_preset = Fetch.get_preset_by_id(preset_id)

    assert code_preset is not None, "Code preset was not fetched correctly"
    all_code_presets = Fetch.get_all_presets_by_category(CodePresetId.CREATE_COLUMNS)
    assert len(all_code_presets) == 1, "The amount of code presets was not equal to 1"

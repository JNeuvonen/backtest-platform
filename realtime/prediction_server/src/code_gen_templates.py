class CodeTemplates:
    FETCH_DATASOURCES = """
{FETCH_DATASOURCES_FUNC}
fetched_data = fetch_datasources() 
"""

    DATA_TRANSFORMATIONS = """
{DATA_TRANSFORMATIONS_FUNC}
transformed_data = make_data_transformations()
"""
    GEN_TRADE_DECISIONS = """
{ENTER_TRADE_FUNC}
{EXIT_TRADE_FUNC}

should_enter_trade = get_enter_trade_decision()
should_exit_trade = get_exit_trade_decision()
"""

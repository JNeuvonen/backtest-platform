package main

type StrategyResponse struct {
	Data []Strategy `json:"data"`
}

type Strategy struct {
	ID                         int     `json:"id"`
	CreatedAt                  string  `json:"created_at"`
	UpdatedAt                  string  `json:"updated_at"`
	Symbol                     string  `json:"symbol"`
	EnterTradeCode             string  `json:"enter_trade_code"`
	ExitTradeCode              string  `json:"exit_trade_code"`
	FetchDatasourcesCode       string  `json:"fetch_datasources_code"`
	DataTransformationsCode    string  `json:"data_transformations_code"`
	Priority                   int     `json:"priority"`
	KlineSizeMs                int     `json:"kline_size_ms"`
	MinimumTimeBetweenTradesMs int     `json:"minimum_time_between_trades_ms"`
	KlinesLeftTillAutoclose    int     `json:"klines_left_till_autoclose"`
	TimeOnTradeOpenMs          int     `json:"time_on_trade_open_ms"`
	PriceOnTradeOpen           float64 `json:"price_on_trade_open"`
	AllocatedSizePerc          float64 `json:"allocated_size_perc"`
	TakeProfitThresholdPerc    float64 `json:"take_profit_threshold_perc"`
	StopLossThresholdPerc      float64 `json:"stop_loss_threshold_perc"`
	UseTimeBasedClose          bool    `json:"use_time_based_close"`
	UseProfitBasedClose        bool    `json:"use_profit_based_close"`
	UseStopLossBasedClose      bool    `json:"use_stop_loss_based_close"`
	UseTakerOrder              bool    `json:"use_taker_order"`
	ShouldEnterTrade           bool    `json:"should_enter_trade"`
	ShouldCloseTrade           bool    `json:"should_close_trade"`
	IsPaperTradeMode           bool    `json:"is_paper_trade_mode"`
	IsLeverageAllowed          bool    `json:"is_leverage_allowed"`
	IsShortSellingStrategy     bool    `json:"is_short_selling_strategy"`
	IsDisabled                 bool    `json:"is_disabled"`
	IsInPosition               bool    `json:"is_in_position"`
}

type CloudLogBody struct {
	Message string `json:"message"`
	Level   string `json:"level"`
}

type SymbolInfoSimple struct {
	Symbol string `json:"symbol"`
	Price  string `json:"price"`
}

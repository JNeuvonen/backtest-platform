package main

type StrategyResponse struct {
	Data []Strategy `json:"data"`
}

type LongShortGroupResponse struct {
	Data []LongShortGroup `json:"data"`
}

type LongShortTickerResponse struct {
	Data []LongShortTicker `json:"data"`
}

type LongShortPairResponse struct {
	Data []LongShortPair `json:"data"`
}

type AccountByNameResponse struct {
	Data Account `json:"data"`
}

type Strategy struct {
	ID                         int     `json:"id"`
	Name                       string  `json:"name"`
	CreatedAt                  string  `json:"created_at"`
	UpdatedAt                  string  `json:"updated_at"`
	Symbol                     string  `json:"symbol"`
	BaseAsset                  string  `json:"base_asset"`
	QuoteAsset                 string  `json:"quote_asset"`
	EnterTradeCode             string  `json:"enter_trade_code"`
	ExitTradeCode              string  `json:"exit_trade_code"`
	FetchDatasourcesCode       string  `json:"fetch_datasources_code"`
	DataTransformationsCode    string  `json:"data_transformations_code"`
	TradeQuantityPrecision     int     `json:"trade_quantity_precision"`
	Priority                   int     `json:"priority"`
	KlineSizeMs                int     `json:"kline_size_ms"`
	MinimumTimeBetweenTradesMs int     `json:"minimum_time_between_trades_ms"`
	MaximumKlinesHoldTime      int     `json:"maximum_klines_hold_time"`
	TimeOnTradeOpenMs          int64   `json:"time_on_trade_open_ms"`
	QuantityOnTradeOpen        float64 `json:"quantity_on_trade_open"`
	RemainingPositionOnTrade   float64 `json:"remaining_position_on_trade"`
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
	IsOnPredServErr            bool    `json:"is_on_pred_serv_err"`
	IsPaperTradeMode           bool    `json:"is_paper_trade_mode"`
	IsLeverageAllowed          bool    `json:"is_leverage_allowed"`
	IsShortSellingStrategy     bool    `json:"is_short_selling_strategy"`
	IsDisabled                 bool    `json:"is_disabled"`
	IsInPosition               bool    `json:"is_in_position"`
}

type LongShortGroup struct {
	ID                       int     `json:"id"`
	Name                     string  `json:"name"`
	CandleInterval           string  `json:"candle_interval"`
	BuyCond                  string  `json:"buy_cond"`
	SellCond                 string  `json:"sell_cond"`
	ExitCond                 string  `json:"exit_cond"`
	NumReqKlines             int     `json:"num_req_klines"`
	MaxSimultaneousPositions int     `json:"max_simultaneous_positions"`
	KlinesUntilClose         int     `json:"klines_until_close"`
	KlineSizeMs              int     `json:"kline_size_ms"`
	MaxLeverageRatio         float64 `json:"max_leverage_ratio"`
	TakeProfitThresholdPerc  float64 `json:"take_profit_threshold_perc"`
	StopLossThresholdPerc    float64 `json:"stop_loss_threshold_perc"`
	IsDisabled               bool    `json:"is_disabled"`
	UseTimeBasedClose        bool    `json:"use_time_based_close"`
	UseProfitBasedClose      bool    `json:"use_profit_based_close"`
	UseStopLossBasedClose    bool    `json:"use_stop_loss_based_close"`
	UseTakerOrder            bool    `json:"use_taker_order"`
}

type LongShortTicker struct {
	ID                     int    `json:"id"`
	LongShortGroupID       int    `json:"long_short_group_id"`
	Symbol                 string `json:"symbol"`
	BaseAsset              string `json:"base_asset"`
	QuoteAsset             string `json:"quote_asset"`
	DatasetName            string `json:"dataset_name"`
	LastKlineOpenTimeSec   int64  `json:"last_kline_open_time_sec"`
	TradeQuantityPrecision int    `json:"trade_quantity_precision"`
	IsValidBuy             bool   `json:"is_valid_buy"`
	IsValidSell            bool   `json:"is_valid_sell"`
	IsOnPredServErr        bool   `json:"is_on_pred_serv_err"`
}

type LongShortPair struct {
	ID                    int     `json:"id"`
	LongShortGroupID      int     `json:"long_short_group_id"`
	BuyTickerID           int     `json:"buy_ticker_id"`
	SellTickerID          int     `json:"sell_ticker_id"`
	BuyTickerDatasetName  string  `json:"buy_ticker_dataset_name"`
	SellTickerDatasetName string  `json:"sell_ticker_dataset_name"`
	BuySymbol             string  `json:"buy_symbol"`
	SellSymbol            string  `json:"sell_symbol"`
	BuyBaseAsset          string  `json:"buy_base_asset"`
	SellBaseAsset         string  `json:"sell_base_asset"`
	BuyQuoteAsset         string  `json:"buy_quote_asset"`
	SellQuoteAsset        string  `json:"sell_quote_asset"`
	BuyOpenTime           int     `json:"buy_open_time_ms"`
	SellOpenTime          int     `json:"sell_open_time_ms"`
	BuyQtyPrecision       int     `json:"buy_qty_precision"`
	SellQtyPrecision      int     `json:"sell_qty_precision"`
	BuyOpenPrice          float64 `json:"buy_open_price"`
	SellOpenPrice         float64 `json:"sell_open_price"`
	BuyOpenQtyInBase      float64 `json:"buy_open_qty_in_base"`
	BuyOpenQtyInQuote     float64 `json:"buy_open_qty_in_quote"`
	SellOpenQtyInQuote    float64 `json:"sell_open_qty_in_quote"`
	DebtOpenQtyInBase     float64 `json:"debt_open_qty_in_base"`
	InPosition            bool    `json:"in_position"`
	ShouldClose           bool    `json:"should_close"`
	IsTradeFinished       bool    `json:"is_trade_finished"`
	ErrorInEntering       bool    `json:"error_in_entering"`
	IsNoLoanAvailableErr  bool    `json:"is_no_loan_available_err"`
}

type CloudLogBody struct {
	Message       string `json:"message"`
	Level         string `json:"level"`
	SourceProgram int32  `json:"source_program"`
}

type SymbolInfoSimple struct {
	Symbol string `json:"symbol"`
	Price  string `json:"price"`
}

type Account struct {
	CreatedAt            string  `json:"created_at"`
	UpdatedAt            string  `json:"updated_at"`
	Name                 string  `json:"name"`
	MaxDebtRatio         float64 `json:"max_debt_ratio"`
	MaxRatioOfLongsToNav float64 `json:"max_ratio_of_longs_to_nav"`
	PreventAllTrading    bool    `json:"prevent_all_trading"`
}

type BodyCreateTrade struct {
	OpenTimeMs              int64   `json:"open_time_ms"`
	StrategyID              int     `json:"strategy_id"`
	Quantity                float64 `json:"quantity"`
	CumulativeQuoteQuantity float64 `json:"cumulative_quote_quantity"`
	OpenPrice               float64 `json:"open_price"`
	Direction               string  `json:"direction"`
}

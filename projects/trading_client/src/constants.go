package main

const (
	DIRECTION_LONG  = "LONG"
	DIRECTION_SHORT = "SHORT"
)

const (
	ASSET_USDT = "USDT"
)

const (
	USDT_QUOTE_BUFFER     = 5
	USDT_MIN_SIZE_FOR_POS = 12
)

const (
	CLOSE_SHORT_FEES_COEFF   = 1.00075
	FEES_REDUCE_AMOUNT_COEFF = 0.99925
)

const (
	ORDER_BUY  = "BUY"
	ORDER_SELL = "SELL"
)

const (
	MINIMUM_USDT_QUANT_FOR_TRADE = 15.0
)

const (
	MARKET_ORDER = "MARKET"
)

const (
	LOG_EXCEPTION = "exception"
	LOG_INFO      = "info"
)

const (
	MINUTE_IN_MS = 1000 * 60
)

const (
	UNABLE_TO_REACH_BE_TRADE_COOLDOWN_MS     = MINUTE_IN_MS * 60 * 4
	FAILED_CALLS_TO_UPDATE_STRAT_STATE_LIMIT = 10
)

type RiskManagementParams struct {
	TradingCooldownStartedTs         int64
	FailedCallsToUpdateStrategyState int32
}

var riskManagementParams *RiskManagementParams // singleton pattern

func GetRiskManagementParams() *RiskManagementParams {
	if riskManagementParams == nil {
		riskManagementParams = &RiskManagementParams{}
		riskManagementParams.TradingCooldownStartedTs = 0
		riskManagementParams.FailedCallsToUpdateStrategyState = 0
		return riskManagementParams
	}
	return riskManagementParams
}

func StartTradingCooldown() {
	riskManagementsParams := GetRiskManagementParams()
	riskManagementsParams.TradingCooldownStartedTs = GetTimeInMs()
}

func IncrementFailedCallsToUpdateStrat() {
	riskManagementsParams := GetRiskManagementParams()
	riskManagementsParams.FailedCallsToUpdateStrategyState += 1
}

func GetNumFailedCallsToPredServer() int32 {
	riskManagementsParams := GetRiskManagementParams()
	return riskManagementsParams.FailedCallsToUpdateStrategyState
}

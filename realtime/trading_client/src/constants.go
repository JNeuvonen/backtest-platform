package main

const (
	TESTNET = "https://testnet.binance.vision"
	MAINNET = "https://api.binance.com"
)

const (
	PRED_SERV_V1_STRAT = "v1/strategy"
	PRED_SERV_V1_LOG   = "v1/log"
	PRED_SERV_V1_ACC   = "v1/acc"
	PRED_SERV_V1_TRADE = "v1/trade"
)

const (
	DIRECTION_LONG  = "LONG"
	DIRECTION_SHORT = "SHORT"
)

const (
	V3_PRICE = "/api/v3/ticker/price"
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

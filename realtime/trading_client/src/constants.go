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
	V3_PRICE = "/api/v3/ticker/price"
)

const (
	TRADING_COOLDOWN_MS = 60000 * 30
)

type RiskManagementParams struct {
	TradingCooldownStartedTs int64
}

var riskManagementsParams *RiskManagementParams // singleton pattern

func GetRiskManagementParams() *RiskManagementParams {
	if riskManagementsParams == nil {
		riskManagementsParams = &RiskManagementParams{}
		riskManagementsParams.TradingCooldownStartedTs = 0
		return riskManagementsParams
	}
	return riskManagementsParams
}

func StartTradingCooldown() {
	riskManagementsParams := GetRiskManagementParams()
	riskManagementsParams.TradingCooldownStartedTs = GetTimeInMs()
}

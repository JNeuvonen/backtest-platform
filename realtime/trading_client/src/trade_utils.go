package main

import (
	"errors"
	"math"

	binance_connector "live/trading-client/src/thirdparty/binance-connector-go"
)

func GetBaseQuantity(sizeUSDT float64, price float64, maxPrecision int32) float64 {
	initialQuantity := sizeUSDT / price

	power := math.Pow(10, float64(maxPrecision))
	adjustedQuantity := math.Floor(initialQuantity*power) / power

	return adjustedQuantity
}

func UpdatePredServerAfterTradeOpen(
	strat Strategy,
	res *binance_connector.MarginAccountNewOrderResponseFULL,
	direction string,
) {
	if res == nil {
		return
	}

	execPrice := SafeDivide(
		ParseToFloat64(res.CumulativeQuoteQty, 0.0),
		ParseToFloat64(res.ExecutedQty, 0.0),
	)

	tradeID := CreateTradeEntry(map[string]interface{}{
		"open_price":                execPrice,
		"open_time_ms":              res.TransactTime,
		"quantity":                  res.ExecutedQty,
		"cumulative_quote_quantity": res.CumulativeQuoteQty,
		"direction":                 direction,
		"strategy_id":               int32(strat.ID),
	})

	updateStratSuccess := UpdateStrategy(map[string]interface{}{
		"id":                         int32(strat.ID),
		"price_on_trade_open":        execPrice,
		"time_on_trade_open_ms":      res.TransactTime,
		"klines_left_till_autoclose": strat.MaximumKlinesHoldTime,
		"active_trade_id":            tradeID,
		"is_in_position":             true,
	})

	if !updateStratSuccess {
		CreateCloudLog(
			NewFmtError(
				errors.New(
					"Updating strategy failed after opening a trade: initiating a trading cooldown as a safety measure.",
				),
				CaptureStack(),
			).Error(),
			"exception",
		)
		StartTradingCooldown()
		IncrementFailedCallsToUpdateStrat()
	}
}

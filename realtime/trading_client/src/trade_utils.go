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

func UpdatePredServerOnTradeClose(
	strat Strategy,
	res *binance_connector.MarginAccountNewOrderResponseFULL,
	direction string,
) {
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
		"open_price":   execPrice,
		"open_time_ms": res.TransactTime,
		"quantity":     res.ExecutedQty,
		"direction":    direction,
		"strategy_id":  int32(strat.ID),
	})

	updateStratSuccess := UpdateStrategy(map[string]interface{}{
		"id":                          int32(strat.ID),
		"price_on_trade_open":         execPrice,
		"quantity_on_trade_open":      res.ExecutedQty,
		"remaining_position_on_trade": res.ExecutedQty,
		"time_on_trade_open_ms":       res.TransactTime,
		"klines_left_till_autoclose":  strat.MaximumKlinesHoldTime,
		"active_trade_id":             tradeID,
		"is_in_position":              true,
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

func GetInterestInAsset(
	marginAssetsRes *binance_connector.CrossMarginAccountDetailResponse,
	asset string,
) float64 {
	if marginAssetsRes != nil {
		for _, item := range marginAssetsRes.UserAssets {
			if item.Asset == asset {
				return ParseToFloat64(item.Interest, 0)
			}
		}
	}
	return 0
}

func GetTotalLiabilitiesInAsset(
	marginAssetsRes *binance_connector.CrossMarginAccountDetailResponse,
	asset string,
) float64 {
	if marginAssetsRes != nil {
		for _, item := range marginAssetsRes.UserAssets {
			if item.Asset == asset {
				return ParseToFloat64(item.Borrowed, 0) + ParseToFloat64(item.Interest, 0)
			}
		}
	}
	return 0.0
}

func GetFreeBalanceForMarginAsset(
	marginAssetsRes *binance_connector.CrossMarginAccountDetailResponse,
	asset string,
) float64 {
	if marginAssetsRes != nil {
		for _, item := range marginAssetsRes.UserAssets {
			if item.Asset == asset {
				return ParseToFloat64(item.Free, 0)
			}
		}
	}
	return 0.0
}

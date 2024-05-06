package main

import (
	"errors"
	"fmt"
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
) {
	if res == nil {
		CreateCloudLog(
			NewFmtError(
				errors.New(
					fmt.Sprintf(
						"%s\n%s: res *binance_connector.MarginAccountNewOrderResponseFULL was unexpectedly nil when closing a trade.",
						strat.Symbol,
						GetCurrentFunctionName(),
					),
				),
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return
	}

	execPrice := SafeDivide(
		ParseToFloat64(res.CumulativeQuoteQty, 0.0),
		ParseToFloat64(res.ExecutedQty, 0.0),
	)

	UpdateStrategyOnTradeClose(strat, map[string]interface{}{
		"quantity":      res.ExecutedQty,
		"price":         execPrice,
		"close_time_ms": res.TransactTime,
	})
}

func UpdatePredServerAfterTradeOpen(
	strat Strategy,
	res *binance_connector.MarginAccountNewOrderResponseFULL,
	direction string,
) {
	if res == nil {
		CreateCloudLog(
			NewFmtError(
				errors.New(
					fmt.Sprintf(
						"%s\n%s: res *binance_connector.MarginAccountNewOrderResponseFULL was unexpectedly nil when opening a trade.",
						strat.Symbol,
						GetCurrentFunctionName(),
					),
				),
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
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
			LOG_EXCEPTION,
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

func GetTotalLongsUSDT(bc *BinanceClient) float64 {
	marginAssetsRes := bc.FetchMarginBalances()
	if marginAssetsRes == nil {
		return 0.0
	}

	usdtPrices, err := getAllUSDTPrices()
	if err != nil {
		return 0.0
	}

	totalLongs := 0.0

	if marginAssetsRes != nil {
		for _, item := range marginAssetsRes.UserAssets {
			freeAsset := ParseToFloat64(item.Free, 0.0)

			if freeAsset > 0.0 {
				if item.Asset == ASSET_USDT {
					continue
				}

				symbolInfo := FindListItem[SymbolInfoSimple](
					usdtPrices,
					func(i SymbolInfoSimple) bool { return i.Symbol == item.Asset+ASSET_USDT },
				)

				if symbolInfo == nil {
					continue
				}

				totalLongs += ParseToFloat64(symbolInfo.Price, 0.0) * freeAsset

			}
		}
	}

	return totalLongs
}

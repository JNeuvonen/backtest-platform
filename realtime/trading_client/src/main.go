package main

func main() {
	tradingConfig := GetTradingConfig()
	NewBinanceClient(tradingConfig)
}

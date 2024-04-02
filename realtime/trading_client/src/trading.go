package main

import (
	"time"
)

func TradingLoop() {
	for {
		strategies, err := FetchStrategies()
		if err != nil {
			time.Sleep(5 * time.Second)
			continue
		}
	}
}

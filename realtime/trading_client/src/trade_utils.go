package main

import (
	"math"
)

func GetBaseQuantity(sizeUSDT float64, price float64, maxPrecision int32) float64 {
	initialQuantity := sizeUSDT / price

	power := math.Pow(10, float64(maxPrecision))
	adjustedQuantity := math.Floor(initialQuantity*power) / power

	return adjustedQuantity
}

package main

import (
	"math"
)

func RoundToPrecision(value float64, precision int32) float64 {
	factor := math.Pow(10, float64(precision))
	return math.Floor(value*factor) / factor
}

func SafeDivide(numerator, denominator float64) float64 {
	if denominator == 0.0 {
		return 0.0
	}
	return numerator / denominator
}

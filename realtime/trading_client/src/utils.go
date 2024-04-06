package main

import (
	"fmt"
	"runtime/debug"
	"strconv"
	"time"
)

func FormatCloudLog(err error, message string, functionName string) string {
	stack := debug.Stack()
	logMessage := fmt.Sprintf(
		"Error in %s: %s - %s. Stack trace: %s",
		functionName,
		message,
		err.Error(),
		stack,
	)
	return logMessage
}

type FmtError struct {
	error
	stack string
}

func CaptureStack() string {
	return string(debug.Stack())
}

func NewFmtError(err error, stack string) *FmtError {
	if err == nil {
		return nil
	}
	return &FmtError{
		error: err,
		stack: stack,
	}
}

func (e *FmtError) Error() string {
	return fmt.Sprintf("%v\nStack Trace:\n%s", e.error, e.stack)
}

func GetTimeInMs() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}

func ParseToFloat64(s string, fallback float64) float64 {
	value, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fallback
	}
	return value
}

func ParseToInt(s string, bitSize int, fallback int64) int64 {
	value, err := strconv.ParseInt(s, 10, bitSize)
	if err != nil {
		return fallback
	}
	return value
}

func ParseToInt8(s string, fallback int8) int8 {
	return int8(ParseToInt(s, 8, int64(fallback)))
}

func ParseToInt16(s string, fallback int16) int16 {
	return int16(ParseToInt(s, 16, int64(fallback)))
}

func ParseToInt32(s string, fallback int32) int32 {
	return int32(ParseToInt(s, 32, int64(fallback)))
}

func ParseToInt64(s string, fallback int64) int64 {
	return ParseToInt(s, 64, fallback)
}

func SafeDivide(numerator, denominator float64) float64 {
	if denominator == 0.0 {
		return 0.0
	}
	return numerator / denominator
}

func FindListItem[T any](items []T, predicate func(T) bool) *T {
	for _, item := range items {
		if predicate(item) {
			return &item
		}
	}
	return nil
}

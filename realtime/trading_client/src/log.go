package main

func LogAndRetFallback[T any](err error, defaultValue T) T {
	CreateCloudLog(
		NewFmtError(err, CaptureStack()).Error(),
		"exception",
	)
	return defaultValue
}

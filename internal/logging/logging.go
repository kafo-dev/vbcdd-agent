// Package logging installs the process logger.
//
// The interactive prompt owns the terminal, so the log goes to a file and a
// debugging session reads it with tail. `VBCDD_AGENT_LOG_FILE=stderr` sends it
// to the terminal instead.
package logging

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"vbcdd.kafo.dev/agent/internal/config"
)

const (
	stderrTarget = "stderr"
	dirMode      = 0o750
	fileMode     = 0o600
)

// Setup installs the default logger and returns it with the release function.
func Setup() (*slog.Logger, func() error, error) {
	level, err := levelFromEnv()
	if err != nil {
		return nil, nil, err
	}
	target := os.Getenv(config.EnvLogFile)
	if target == "" {
		path, pathErr := config.LogPath()
		if pathErr != nil {
			return nil, nil, pathErr
		}
		target = path
	}
	if target == stderrTarget {
		logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
		slog.SetDefault(logger)
		return logger, func() error { return nil }, nil
	}
	dir := filepath.Dir(target)
	if err = os.MkdirAll(dir, dirMode); err != nil { //nolint:gosec // G703: the log path comes from the config.
		return nil, nil, fmt.Errorf("logging: create directory: %w", err)
	}
	const openFlags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	file, openErr := os.OpenFile(target, openFlags, fileMode) //nolint:gosec // G703: the path comes from the config.
	if openErr != nil {
		return nil, nil, fmt.Errorf("logging: open %s: %w", target, openErr)
	}
	logger := slog.New(slog.NewTextHandler(file, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)
	return logger, file.Close, nil
}

// levelFromEnv reads `VBCDD_AGENT_LOG_LEVEL`, defaulting to `info`.
func levelFromEnv() (slog.Level, error) {
	raw := os.Getenv(config.EnvLogLevel)
	if raw == "" {
		return slog.LevelInfo, nil
	}
	var level slog.Level
	if err := level.UnmarshalText([]byte(raw)); err != nil {
		return 0, fmt.Errorf("logging: %s=%q: %w", config.EnvLogLevel, raw, err)
	}
	return level, nil
}

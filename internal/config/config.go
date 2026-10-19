// Package config reads the on-disk configuration and resolves the paths the
// program writes to.
//
// Configuration lives below the XDG configuration root and transcripts below
// the XDG data root, so a delete rebuilds one but loses the other.
package config

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// Environment variables. Every name carries the application prefix, and every
// path stays overridable so a test run touches no real data.
const (
	EnvConfigDir = "VBCDD_AGENT_CONFIG_DIR"
	EnvDataDir   = "VBCDD_AGENT_DATA_DIR"
	EnvLogLevel  = "VBCDD_AGENT_LOG_LEVEL"
	EnvLogFile   = "VBCDD_AGENT_LOG_FILE"
	EnvBaseURL   = "VBCDD_AGENT_BASE_URL"
	EnvAPIKey    = "VBCDD_AGENT_API_KEY" //nolint:gosec // G101: the name of an environment variable, not a credential.
	EnvModel     = "VBCDD_AGENT_MODEL"
)

const (
	// dirName is the directory the program owns below each XDG base
	// directory, from the configuration root to the data root.
	dirName        = "vbcdd/agent"
	configName     = "config.toml"
	sessionDirName = "sessions"
	logName        = "log.txt"

	// DefaultBaseURL speaks the OpenAI chat-completions protocol.
	DefaultBaseURL = "https://api.openai.com/v1"
	// DefaultModel is a small, widely available tool-calling model.
	DefaultModel = "gpt-4o-mini"
	// DefaultMaxTurns bounds the tool loop of a single request.
	DefaultMaxTurns = 20
	// DefaultShellTimeoutSeconds bounds one shell command.
	DefaultShellTimeoutSeconds = 60
	// DefaultMaxOutputBytes caps the output one shell command returns.
	DefaultMaxOutputBytes = 64 << 10

	// keyCommandTimeout bounds the command that prints the API key.
	keyCommandTimeout = 15 * time.Second
	// keyCommandMaxOutput caps what that command may print.
	keyCommandMaxOutput = 8 << 10
)

// Config is the whole file. An empty field falls back to its default.
type Config struct {
	BaseURL             string            `toml:"base_url"`
	APIKey              string            `toml:"api_key"`
	APIKeyCommand       string            `toml:"api_key_command"`
	Model               string            `toml:"model"`
	ReasoningEffort     string            `toml:"reasoning_effort"`
	SystemPrompt        string            `toml:"system_prompt"`
	SystemPromptFile    string            `toml:"system_prompt_file"`
	Headers             map[string]string `toml:"headers"`
	MaxTurns            int               `toml:"max_turns"`
	ShellTimeoutSeconds int               `toml:"shell_timeout_seconds"`
	MaxOutputBytes      int               `toml:"max_output_bytes"`

	// Path records the file the configuration came from, empty when the
	// program runs on defaults.
	Path string `toml:"-"`
}

// configDir returns the directory holding the configuration file. A missing
// directory is not an error; the caller writes it on demand.
func configDir() (string, error) {
	if override := os.Getenv(EnvConfigDir); override != "" {
		return override, nil
	}
	base, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("config: user config dir: %w", err)
	}
	return filepath.Join(base, dirName), nil
}

// dataDir returns the directory holding transcripts and the log.
func dataDir() (string, error) {
	if override := os.Getenv(EnvDataDir); override != "" {
		return override, nil
	}
	base, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("config: user home dir: %w", err)
	}
	if xdg := os.Getenv("XDG_DATA_HOME"); xdg != "" {
		base = xdg
	} else {
		base = filepath.Join(base, ".local", "share")
	}
	return filepath.Join(base, dirName), nil
}

// Path returns the configuration file path.
func Path() (string, error) {
	dir, err := configDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, configName), nil
}

// SessionDir returns the directory holding saved conversations.
func SessionDir() (string, error) {
	dir, err := dataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, sessionDirName), nil
}

// LogPath returns the log file path.
func LogPath() (string, error) {
	dir, err := dataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, logName), nil
}

// Load reads the configuration, then applies the environment overrides and the
// defaults. A missing file yields the defaults.
func Load() (Config, error) {
	cfg := Config{}
	path, err := Path()
	if err != nil {
		return Config{}, err
	}
	raw, err := os.ReadFile(path)
	if err == nil {
		cfg.Path = path
		if err = toml.Unmarshal(raw, &cfg); err != nil {
			return Config{}, fmt.Errorf("config: %s: %w", path, err)
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return Config{}, fmt.Errorf("config: %s: %w", path, err)
	}
	cfg.applyEnv()
	cfg.applyDefaults()
	if err = cfg.resolvePrompt(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// applyEnv copies the environment overrides over the file.
func (c *Config) applyEnv() {
	if v := os.Getenv(EnvBaseURL); v != "" {
		c.BaseURL = v
	}
	if v := os.Getenv(EnvAPIKey); v != "" {
		c.APIKey = v
	}
	if v := os.Getenv(EnvModel); v != "" {
		c.Model = v
	}
}

// applyDefaults fills every empty field. One config table is the single source
// of truth: the code reads a value here, never a second copy.
func (c *Config) applyDefaults() {
	if c.BaseURL == "" {
		c.BaseURL = DefaultBaseURL
	}
	if c.Model == "" {
		c.Model = DefaultModel
	}
	if c.MaxTurns <= 0 {
		c.MaxTurns = DefaultMaxTurns
	}
	if c.ShellTimeoutSeconds <= 0 {
		c.ShellTimeoutSeconds = DefaultShellTimeoutSeconds
	}
	if c.MaxOutputBytes <= 0 {
		c.MaxOutputBytes = DefaultMaxOutputBytes
	}
}

// resolvePrompt loads `system_prompt_file` into `system_prompt` when the file
// stands in for the inline text.
func (c *Config) resolvePrompt() error {
	if c.SystemPromptFile == "" {
		return nil
	}
	path := expandHome(c.SystemPromptFile)
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("config: system prompt file: %w", err)
	}
	c.SystemPrompt = string(raw)
	return nil
}

// Validate reports a configuration the client cannot use.
func (c *Config) Validate() error {
	if strings.TrimSpace(c.BaseURL) == "" {
		return errors.New("config: base_url is empty")
	}
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("config: model is empty")
	}
	if strings.TrimSpace(c.APIKey) == "" {
		return errors.New("config: api_key is empty: set it in the config file or in " + EnvAPIKey)
	}
	return nil
}

// ResolveAPIKey runs `api_key_command` and stores its output as the key. A
// configured command wins over the inline key, because it carries the fresher
// credential. An empty command leaves the key as it stands.
func (c *Config) ResolveAPIKey() error {
	if strings.TrimSpace(c.APIKeyCommand) == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), keyCommandTimeout)
	defer cancel()

	shellName := os.Getenv("SHELL")
	if shellName == "" {
		shellName = "/bin/sh"
	}
	//nolint:gosec // G204: the user configured this command to print the key.
	cmd := exec.CommandContext(ctx, shellName, "-c", c.APIKeyCommand)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &limitedWriter{target: &stdout, max: keyCommandMaxOutput}
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("config: %s: %w: %s", c.APIKeyCommand, err, strings.TrimSpace(stderr.String()))
	}
	key := strings.TrimSpace(stdout.String())
	if key == "" {
		return fmt.Errorf("config: %s printed no key", c.APIKeyCommand)
	}
	c.APIKey = key
	return nil
}

// limitedWriter collects at most max bytes and drops the rest.
type limitedWriter struct {
	target *bytes.Buffer
	max    int
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	written := len(p)
	if room := w.max - w.target.Len(); room > 0 {
		if len(p) > room {
			p = p[:room]
		}
		w.target.Write(p)
	}
	return written, nil
}

// expandHome resolves a leading `~` against the user's home directory.
func expandHome(path string) string {
	if !strings.HasPrefix(path, "~") {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, strings.TrimPrefix(path, "~"))
}

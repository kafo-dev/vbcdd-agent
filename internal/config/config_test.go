package config

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAppliesDefaults(t *testing.T) {
	t.Setenv(EnvConfigDir, t.TempDir())
	t.Setenv(EnvDataDir, t.TempDir())

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.BaseURL != DefaultBaseURL {
		t.Fatalf("base url = %q, want %q", cfg.BaseURL, DefaultBaseURL)
	}
	if cfg.Model != DefaultModel {
		t.Fatalf("model = %q, want %q", cfg.Model, DefaultModel)
	}
	if cfg.MaxTurns != DefaultMaxTurns {
		t.Fatalf("max turns = %d, want %d", cfg.MaxTurns, DefaultMaxTurns)
	}
}

func TestLoadReadsFileAndEnvironment(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvConfigDir, dir)
	t.Setenv(EnvDataDir, t.TempDir())
	body := "base_url = \"https://example.test/v1\"\napi_key = \"file-key\"\nmodel = \"file-model\"\n"
	if err := os.WriteFile(filepath.Join(dir, configName), []byte(body), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	t.Setenv(EnvModel, "env-model")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.BaseURL != "https://example.test/v1" {
		t.Fatalf("base url = %q", cfg.BaseURL)
	}
	if cfg.APIKey != "file-key" {
		t.Fatalf("api key = %q", cfg.APIKey)
	}
	if cfg.Model != "env-model" {
		t.Fatalf("model = %q, want the environment override", cfg.Model)
	}
}

func TestLoadReadsPromptFile(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvConfigDir, dir)
	t.Setenv(EnvDataDir, t.TempDir())
	prompt := filepath.Join(dir, "prompt.md")
	if err := os.WriteFile(prompt, []byte("stay terse"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	body := fmt.Sprintf("system_prompt_file = %q\n", prompt)
	if err := os.WriteFile(filepath.Join(dir, configName), []byte(body), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.SystemPrompt != "stay terse" {
		t.Fatalf("system prompt = %q, want the file contents", cfg.SystemPrompt)
	}
}

func TestValidateRejectsMissingKey(t *testing.T) {
	t.Parallel()
	cfg := Config{BaseURL: DefaultBaseURL, Model: DefaultModel}
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate accepted a missing api key")
	}
}

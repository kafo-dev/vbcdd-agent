package shell

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestRunCapturesOutputAndExitCode(t *testing.T) {
	t.Parallel()
	runner := Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}

	res, err := runner.Run(context.Background(), "printf 'hello'")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Output != "hello" {
		t.Fatalf("output = %q, want %q", res.Output, "hello")
	}
	if res.ExitCode != 0 {
		t.Fatalf("exit code = %d, want 0", res.ExitCode)
	}

	failed, err := runner.Run(context.Background(), "exit 3")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if failed.ExitCode != 3 {
		t.Fatalf("exit code = %d, want 3", failed.ExitCode)
	}
}

func TestRunUsesWorkingDirectory(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	runner := Runner{Shell: "/bin/sh", Dir: dir, Timeout: 5 * time.Second, MaxOutputBytes: 4096}

	res, err := runner.Run(context.Background(), "pwd")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.TrimSpace(res.Output) != dir {
		t.Fatalf("pwd = %q, want %q", strings.TrimSpace(res.Output), dir)
	}
}

func TestRunTimesOut(t *testing.T) {
	t.Parallel()
	runner := Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 100 * time.Millisecond, MaxOutputBytes: 4096}

	res, err := runner.Run(context.Background(), "sleep 5")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !res.TimedOut {
		t.Fatal("TimedOut = false, want true")
	}
}

func TestRunTruncatesOutput(t *testing.T) {
	t.Parallel()
	runner := Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 8}

	res, err := runner.Run(context.Background(), "printf '0123456789abcdef'")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.HasPrefix(res.Output, "01234567") {
		t.Fatalf("output = %q, want it to start with the first 8 bytes", res.Output)
	}
	if !strings.Contains(res.Output, "truncated") {
		t.Fatalf("output = %q, want a truncation marker", res.Output)
	}
}

func TestRunRejectsEmptyCommand(t *testing.T) {
	t.Parallel()
	runner := Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: time.Second, MaxOutputBytes: 4096}
	if _, err := runner.Run(context.Background(), "  "); err == nil {
		t.Fatal("Run accepted an empty command")
	}
}

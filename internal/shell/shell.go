// Package shell runs one command through the user's system shell.
package shell

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// fallbackShell runs the command when `$SHELL` names nothing.
const fallbackShell = "/bin/sh"

// Runner executes commands with a working directory, a timeout, and an output
// cap.
type Runner struct {
	// Shell is the interpreter. Its empty value reads `$SHELL`.
	Shell          string
	Dir            string
	Timeout        time.Duration
	MaxOutputBytes int
}

// Result is what one command produced.
type Result struct {
	Output   string
	ExitCode int
	TimedOut bool
}

// New resolves the interpreter and fills the bounds.
func New(shellName, dir string, timeout time.Duration, maxOutputBytes int) Runner {
	if shellName == "" {
		shellName = os.Getenv("SHELL")
	}
	if shellName == "" {
		shellName = fallbackShell
	}
	return Runner{Shell: shellName, Dir: dir, Timeout: timeout, MaxOutputBytes: maxOutputBytes}
}

// Run executes command and returns the combined output with the exit code.
// A non-zero exit is a result, not an error; only a failure to start or a
// deadline returns one.
func (r Runner) Run(parent context.Context, command string) (Result, error) {
	if strings.TrimSpace(command) == "" {
		return Result{}, fmt.Errorf("shell: empty command")
	}
	if r.Shell == "" {
		return Result{}, fmt.Errorf("shell: no interpreter")
	}
	if r.Timeout <= 0 {
		return Result{}, fmt.Errorf("shell: no timeout")
	}
	ctx, cancel := context.WithTimeout(parent, r.Timeout)
	defer cancel()

	//nolint:gosec // G204: running the user's command is the tool's purpose.
	cmd := exec.CommandContext(
		ctx,
		r.Shell,
		"-c",
		command,
	)
	cmd.Dir = r.Dir
	cmd.Stdin = nil
	buf := &limitedBuffer{max: r.MaxOutputBytes}
	cmd.Stdout = buf
	cmd.Stderr = buf

	runErr := cmd.Run()
	result := Result{Output: buf.String(), ExitCode: cmd.ProcessState.ExitCode()}
	if ctx.Err() != nil {
		result.TimedOut = true
		result.Output = strings.TrimSpace(result.Output)
		return result, nil
	}
	if runErr == nil {
		return result, nil
	}
	exitErr, isExit := errors.AsType[*exec.ExitError](runErr)
	if isExit {
		result.ExitCode = exitErr.ExitCode()
		return result, nil
	}
	return Result{}, fmt.Errorf("shell: start %s: %w", r.Shell, runErr)
}

// SystemPrompt states the interpreter and the working directory, so the model
// writes a command for the environment it meets.
func (r Runner) SystemPrompt() string {
	return fmt.Sprintf("Commands run through `%s -c` in `%s`.", r.Shell, r.Dir)
}

// limitedBuffer collects at most max bytes and notes the overflow.
type limitedBuffer struct {
	buf       bytes.Buffer
	max       int
	truncated bool
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	written := len(p)
	if b.max <= 0 {
		return written, nil
	}
	room := b.max - b.buf.Len()
	if room <= 0 {
		b.truncated = true
		return written, nil
	}
	if len(p) > room {
		b.truncated = true
		p = p[:room]
	}
	b.buf.Write(p)
	return written, nil
}

func (b *limitedBuffer) String() string {
	if !b.truncated {
		return b.buf.String()
	}
	return b.buf.String() + "\n[output truncated]"
}

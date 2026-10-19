// Command vbcdd-agent is a terminal agent that answers by running shell
// commands through an OpenAI-compatible chat endpoint.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"vbcdd.kafo.dev/agent/internal/agent"
	"vbcdd.kafo.dev/agent/internal/config"
	"vbcdd.kafo.dev/agent/internal/llm"
	"vbcdd.kafo.dev/agent/internal/logging"
	"vbcdd.kafo.dev/agent/internal/shell"
	"vbcdd.kafo.dev/agent/internal/transcript"
)

const (
	prompt     = "agent(%d)> "
	quitTokens = "quit,exit"
	// scannerInit is the starting buffer for one line of input.
	scannerInit = 64 << 10
	// scannerMax bounds one line of input.
	scannerMax = 1 << 20
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "vbcdd-agent: %v\n", err)
		os.Exit(1)
	}
}

type options struct {
	session string
	cont    bool
	list    bool
	once    bool
}

func run() error {
	opts := parseFlags()
	storeDir, err := config.SessionDir()
	if err != nil {
		return err
	}
	store := transcript.Store{Dir: storeDir}
	if opts.list {
		return listSessions(store)
	}

	logger, closeLog, err := logging.Setup()
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := closeLog(); closeErr != nil {
			fmt.Fprintf(os.Stderr, "vbcdd-agent: %v\n", closeErr)
		}
	}()
	return serve(opts, store, logger)
}

// parseFlags reads the command line. Configuration comes from the file alone,
// so the flags name only what a run does, never what it talks to.
func parseFlags() options {
	var opts options
	flag.StringVar(&opts.session, "session", "", "resume the conversation with this id")
	flag.BoolVar(&opts.cont, "continue", false, "resume the most recent conversation")
	flag.BoolVar(&opts.list, "list", false, "list saved conversations and exit")
	flag.BoolVar(&opts.once, "once", false, "answer one message from the arguments or standard input, then exit")
	flag.Parse()
	return opts
}

// serve builds the engine and answers until the conversation ends.
func serve(opts options, store transcript.Store, logger *slog.Logger) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	if err = cfg.Validate(); err != nil {
		return err
	}
	logger.Debug("configuration resolved", "path", cfg.Path, "base_url", cfg.BaseURL, "model", cfg.Model)

	dir, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("main: working directory: %w", err)
	}
	runner := shell.New("", dir, time.Duration(cfg.ShellTimeoutSeconds)*time.Second, cfg.MaxOutputBytes)
	client := llm.New(cfg.BaseURL, cfg.APIKey, cfg.Model, cfg.Headers, nil)

	writer, resumed, err := openTranscript(store, opts)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := writer.Close(); closeErr != nil {
			logger.Error("close transcript", "error", closeErr)
		}
	}()

	ag := agent.New(client, runner, cfg.SystemPrompt, cfg.MaxTurns, os.Stdout, os.Stderr)
	if resumed != nil {
		ag.Restore(resumed)
	}
	saved := len(ag.History())

	message := strings.Join(flag.Args(), " ")
	switch {
	case message != "":
		return oneShot(ag, writer, &saved, logger, message)
	case opts.once || !isTerminal(os.Stdin):
		raw, readErr := io.ReadAll(io.LimitReader(os.Stdin, scannerMax))
		if readErr != nil {
			return fmt.Errorf("main: read standard input: %w", readErr)
		}
		return oneShot(ag, writer, &saved, logger, string(raw))
	default:
		return repl(ag, writer, &saved, logger, sessionLabel(writer, opts))
	}
}

// loadConfig reads the configuration file and resolves the API key. No flag
// shadows the file; it is the source of truth for what the client talks to.
func loadConfig() (config.Config, error) {
	cfg, err := config.Load()
	if err != nil {
		return config.Config{}, err
	}
	if err = cfg.ResolveAPIKey(); err != nil {
		return config.Config{}, err
	}
	return cfg, nil
}

// openTranscript resumes a conversation or starts one.
func openTranscript(store transcript.Store, opts options) (*transcript.Writer, []llm.Message, error) {
	switch {
	case opts.session != "":
		session, messages, err := store.Open(opts.session)
		if err != nil {
			return nil, nil, err
		}
		writer, err := store.Append(session)
		if err != nil {
			return nil, nil, err
		}
		return writer, messages, nil
	case opts.cont:
		session, ok, err := store.Latest()
		if err != nil {
			return nil, nil, err
		}
		if !ok {
			writer, createErr := store.Create()
			if createErr != nil {
				return nil, nil, createErr
			}
			return writer, nil, nil
		}
		messages, readErr := transcript.Read(session.Path)
		if readErr != nil {
			return nil, nil, readErr
		}
		writer, appendErr := store.Append(session)
		if appendErr != nil {
			return nil, nil, appendErr
		}
		return writer, messages, nil
	default:
		writer, err := store.Create()
		if err != nil {
			return nil, nil, err
		}
		return writer, nil, nil
	}
}

// repl reads messages from the terminal until the user leaves.
func repl(
	ag *agent.Agent,
	writer *transcript.Writer,
	saved *int,
	logger *slog.Logger,
	label string,
) error {
	fmt.Fprintf(os.Stderr, "vbcdd-agent — session %s (Ctrl-D or %q to leave)\n", label, quitTokens)
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, scannerInit), scannerMax)
	for {
		fmt.Fprintf(os.Stderr, prompt, ag.Tokens())
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "quit" || line == "exit" {
			break
		}
		if err := answer(ag, writer, saved, logger, line); err != nil {
			fmt.Fprintf(os.Stderr, "vbcdd-agent: %v\n", err)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("main: read input: %w", err)
	}
	fmt.Fprintln(os.Stderr)
	return nil
}

// oneShot answers one message and returns.
func oneShot(ag *agent.Agent, writer *transcript.Writer, saved *int, logger *slog.Logger, message string) error {
	if strings.TrimSpace(message) == "" {
		return errors.New("main: no message given")
	}
	return answer(ag, writer, saved, logger, strings.TrimSpace(message))
}

// answer runs one turn, which streams the reply to standard output, then
// appends the new messages to the transcript.
func answer(ag *agent.Agent, writer *transcript.Writer, saved *int, logger *slog.Logger, message string) error {
	if _, err := ag.Ask(context.Background(), message); err != nil {
		return err
	}
	fmt.Fprintln(os.Stdout)
	return save(writer, ag, saved, logger)
}

// save appends every message the turn added.
func save(writer *transcript.Writer, ag *agent.Agent, saved *int, logger *slog.Logger) error {
	history := ag.History()
	for _, message := range history[*saved:] {
		if err := writer.Append(message); err != nil {
			logger.Error("append transcript", "error", err)
		}
	}
	*saved = len(history)
	return nil
}

// listSessions prints the saved conversations.
func listSessions(store transcript.Store) error {
	sessions, err := store.List()
	if err != nil {
		return err
	}
	if len(sessions) == 0 {
		fmt.Fprintln(os.Stderr, "no saved conversations")
		return nil
	}
	for _, session := range sessions {
		when := session.Started.Format(time.RFC3339)
		fmt.Fprintf(os.Stdout, "%s  %s\n", session.ID, when)
	}
	return nil
}

// sessionLabel names the conversation in the banner.
func sessionLabel(writer *transcript.Writer, opts options) string {
	if opts.session != "" {
		return opts.session
	}
	return writer.Session.ID
}

// isTerminal reports whether file refers to a terminal.
func isTerminal(file *os.File) bool {
	info, err := file.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

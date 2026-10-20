// Package agent drives the tool loop between the model and the shell through
// the Charm Fantasy SDK.
//
// Fantasy owns the loop, the provider translation, the streaming, and the
// retries. This package owns the one tool, the section formatting of the
// streamed output, and the conversation it hands back and forth.
package agent

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"charm.land/fantasy"

	"vbcdd.kafo.dev/agent/internal/shell"
)

// shellToolName is the one tool the model may call.
const shellToolName = "shell"

// charsPerToken is the rule of thumb for an endpoint that reports no usage.
const charsPerToken = 4

// section names the part of the output a streamed fragment belongs to, so one
// section can be kept apart from the next.
type section int

// The sections of a turn, in the order they arrive.
const (
	sectionNone section = iota
	sectionReasoning
	sectionCommand
	sectionOutput
	sectionAnswer
)

// shellInput is the shell tool's argument schema. The SDK generates the JSON
// schema the model reads from the field.
type shellInput struct {
	Cmd string `json:"cmd"`
}

// Config wires one conversation.
type Config struct {
	// Model answers every request.
	Model fantasy.LanguageModel
	// Runner executes the commands the model asks for.
	Runner shell.Runner
	// SystemPrompt stands in front of the conversation; empty leaves it out.
	SystemPrompt string
	// ProviderOptions carries the endpoint-specific settings, such as the
	// reasoning effort. A nil value sends none.
	ProviderOptions fantasy.ProviderOptions
	// MaxTurns bounds the tool loop of one request.
	MaxTurns int
	// Out receives the assistant's text as the model streams it.
	Out io.Writer
	// Thinking receives the model's reasoning as it streams it, before the
	// answer.
	Thinking io.Writer
	// Notices receives each command and the output it prints.
	Notices io.Writer
}

// Agent holds one conversation and the engine that advances it.
type Agent struct {
	engine   fantasy.Agent
	runner   shell.Runner
	maxTurns int
	out      io.Writer
	thinking io.Writer
	notices  io.Writer
	history  []fantasy.Message
	tokens   int
	// open is the writer whose last fragment left its line unfinished, and
	// openSection names the part of the output that fragment belongs to.
	// Together they keep the reasoning, the commands, what those commands
	// print, and the answer from running together.
	open        io.Writer
	openSection section
}

// New starts a conversation.
func New(cfg Config) *Agent {
	maxTurns := cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 1
	}
	agent := &Agent{
		runner:   cfg.Runner,
		maxTurns: maxTurns,
		out:      cfg.Out,
		thinking: cfg.Thinking,
		notices:  cfg.Notices,
	}
	agent.engine = fantasy.NewAgent(
		cfg.Model,
		fantasy.WithSystemPrompt(cfg.SystemPrompt),
		fantasy.WithTools(agent.tool()),
		fantasy.WithProviderOptions(cfg.ProviderOptions),
	)
	return agent
}

// History returns the conversation the transcript stores. The system prompt
// stays out; the engine adds it to every request.
func (a *Agent) History() []fantasy.Message {
	if len(a.history) == 0 {
		return nil
	}
	return append([]fantasy.Message(nil), a.history...)
}

// Restore seeds the conversation from a saved transcript.
func (a *Agent) Restore(messages []fantasy.Message) {
	a.history = append([]fantasy.Message(nil), messages...)
	a.tokens = estimateTokens(a.history)
}

// Tokens reports the size of the context the last request sent, in tokens.
func (a *Agent) Tokens() int {
	return a.tokens
}

// Ask appends input and runs the tool loop until the model answers without a
// tool call. It returns the assistant's final text.
func (a *Agent) Ask(ctx context.Context, input string) (string, error) {
	if strings.TrimSpace(input) == "" {
		return "", fmt.Errorf("agent: empty input")
	}
	a.open, a.openSection = nil, sectionNone
	reply, err := a.run(ctx, input)
	if endErr := a.endLine(); endErr != nil && err == nil {
		err = endErr
	}
	return reply, err
}

// run streams one turn and folds the messages it produced into the history.
func (a *Agent) run(ctx context.Context, input string) (string, error) {
	call := fantasy.AgentStreamCall{
		Prompt:           input,
		Messages:         a.history,
		StopWhen:         []fantasy.StopCondition{fantasy.StepCountIs(a.maxTurns)},
		OnTextDelta:      a.writeText,
		OnReasoningDelta: a.writeReasoning,
	}
	result, err := a.engine.Stream(ctx, call)
	if err != nil {
		return "", fmt.Errorf("agent: stream: %w", err)
	}
	a.history = append(a.history, fantasy.NewUserMessage(input))
	for _, step := range result.Steps {
		a.history = append(a.history, step.Messages...)
	}
	a.recordUsage(result.Steps)
	return result.Response.Content.Text(), nil
}

// recordUsage stores the context size the last step reported, or an estimate
// from the text when it reported none.
func (a *Agent) recordUsage(steps []fantasy.StepResult) {
	if len(steps) > 0 {
		usage := steps[len(steps)-1].Usage
		total := usage.TotalTokens
		if total <= 0 {
			total = usage.InputTokens + usage.OutputTokens
		}
		if total > 0 {
			a.tokens = int(total)
			return
		}
	}
	a.tokens = estimateTokens(a.history)
}

// estimateTokens applies the rule of thumb of one token per `charsPerToken`
// characters.
func estimateTokens(messages []fantasy.Message) int {
	chars := 0
	for _, message := range messages {
		for _, part := range message.Content {
			chars += messagePartChars(part)
		}
	}
	return (chars + charsPerToken - 1) / charsPerToken
}

// messagePartChars counts the characters one message part contributes. Binary
// and reference parts add nothing the endpoint bills for.
func messagePartChars(part fantasy.MessagePart) int {
	switch part.GetType() {
	case fantasy.ContentTypeText:
		text, ok := fantasy.AsMessagePart[fantasy.TextPart](part)
		if !ok {
			return 0
		}
		return len(text.Text)
	case fantasy.ContentTypeReasoning:
		reasoning, ok := fantasy.AsMessagePart[fantasy.ReasoningPart](part)
		if !ok {
			return 0
		}
		return len(reasoning.Text)
	case fantasy.ContentTypeToolCall:
		call, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part)
		if !ok {
			return 0
		}
		return len(call.ToolName) + len(call.Input)
	case fantasy.ContentTypeToolResult:
		result, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part)
		if !ok {
			return 0
		}
		return toolResultChars(result.Output)
	case fantasy.ContentTypeFile, fantasy.ContentTypeSource:
		return 0
	}
	return 0
}

// toolResultChars counts the characters of a tool result's output.
func toolResultChars(output fantasy.ToolResultOutputContent) int {
	if output == nil {
		return 0
	}
	text, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](output)
	if !ok {
		return 0
	}
	return len(text.Text)
}

// writeText sends one streamed fragment of the answer to the output writer.
func (a *Agent) writeText(_ string, text string) error {
	return a.writeTo(a.out, sectionAnswer, text)
}

// writeReasoning sends one reasoning fragment to the thinking writer.
func (a *Agent) writeReasoning(_ string, text string) error {
	return a.writeTo(a.thinking, sectionReasoning, text)
}

// writeTo opens its section when a different one left a line unfinished, then
// sends text to its writer. It remembers the section whose line the text
// leaves unfinished.
func (a *Agent) writeTo(w io.Writer, where section, text string) error {
	if w == nil || text == "" {
		return nil
	}
	if err := a.begin(where); err != nil {
		return err
	}
	if _, err := io.WriteString(w, text); err != nil {
		return fmt.Errorf("agent: write streamed text: %w", err)
	}
	if strings.HasSuffix(text, "\n") {
		a.open, a.openSection = nil, sectionNone
		return nil
	}
	a.open, a.openSection = w, where
	return nil
}

// begin ends the line a different section left unfinished, so the section
// about to start does not run into it.
func (a *Agent) begin(next section) error {
	if a.openSection == next {
		return nil
	}
	return a.endLine()
}

// endLine finishes the line a section left unfinished, if any.
func (a *Agent) endLine() error {
	if a.open == nil {
		return nil
	}
	w, where := a.open, a.openSection
	a.open, a.openSection = nil, sectionNone
	return a.writeTo(w, where, "\n")
}

// tool advertises the shell tool.
func (a *Agent) tool() fantasy.AgentTool {
	return fantasy.NewAgentTool(shellToolName, a.toolDescription(), a.runShell)
}

// toolDescription names the interpreter and the working directory, so the
// model writes a command for the environment it meets.
func (a *Agent) toolDescription() string {
	return fmt.Sprintf(
		"Run a command line through `%s -c` in `%s` and return its combined output.",
		a.runner.Shell,
		abbreviateHome(a.runner.Dir),
	)
}

// runShell executes one requested command and renders its result as text.
func (a *Agent) runShell(
	ctx context.Context,
	input shellInput,
	_ fantasy.ToolCall,
) (fantasy.ToolResponse, error) {
	if strings.TrimSpace(input.Cmd) == "" {
		return fantasy.NewTextErrorResponse("the cmd argument is required"), nil
	}
	if err := a.notice(input.Cmd); err != nil {
		return fantasy.ToolResponse{}, err
	}
	res, err := a.runner.Run(ctx, input.Cmd)
	if err != nil {
		return fantasy.NewTextErrorResponse(err.Error()), nil
	}
	text := render(res)
	if err = a.output(text); err != nil {
		return fantasy.ToolResponse{}, err
	}
	return fantasy.NewTextResponse(text), nil
}

// notice reports a command before it runs.
func (a *Agent) notice(command string) error {
	return a.writeTo(a.notices, sectionCommand, "$ "+command+"\n")
}

// output shows what a command printed, so the run shows the work and not only
// the request.
func (a *Agent) output(text string) error {
	return a.writeTo(a.notices, sectionOutput, text)
}

// render turns a result into the text the model reads next.
func render(res shell.Result) string {
	var sb strings.Builder
	if strings.TrimSpace(res.Output) == "" {
		sb.WriteString("(no output)")
	} else {
		sb.WriteString(res.Output)
	}
	if res.TimedOut {
		sb.WriteString("\n[command timed out]")
	}
	fmt.Fprintf(&sb, "\n[exit code %d]", res.ExitCode)
	return sb.String()
}

// abbreviateHome replaces a leading home directory with `~`, so the model
// meets a short, familiar path instead of an absolute one. Paths outside the
// home directory and an unresolved home are returned unchanged.
func abbreviateHome(path string) string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return path
	}
	if path == home {
		return "~"
	}
	prefix := home + string(os.PathSeparator)
	if strings.HasPrefix(path, prefix) {
		return "~" + path[len(home):]
	}
	return path
}

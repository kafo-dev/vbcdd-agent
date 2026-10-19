// Package agent drives the tool loop between the model and the shell.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"vbcdd.kafo.dev/agent/internal/llm"
	"vbcdd.kafo.dev/agent/internal/shell"
)

// shellToolName is the one tool the model may call.
const shellToolName = "shell"

// toolParameters describes the shell tool's arguments.
const toolParameters = `{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "A command line for the system shell. Pipes, redirection, and globs work because the interpreter expands them."
    }
  },
  "required": ["command"],
  "additionalProperties": false
}`

// section names the part of the output a streamed fragment belongs to, so
// one section can be kept apart from the next.
type section int

// The sections of a turn, in the order they arrive.
const (
	sectionNone section = iota
	sectionReasoning
	sectionCommand
	sectionOutput
	sectionAnswer
)

// Agent holds one conversation and the loop that advances it.
type Agent struct {
	Client       *llm.Client
	Runner       shell.Runner
	SystemPrompt string
	MaxTurns     int
	// Out receives the assistant's text as the model streams it.
	Out io.Writer
	// Thinking receives the model's reasoning as it streams it, before the
	// answer.
	Thinking io.Writer
	// Notices receives each command and the output it prints.
	Notices io.Writer

	history  []llm.Message
	tokens   int
	writeErr error
	// open is the writer whose last fragment left its line unfinished, and
	// openSection names the part of the output that fragment belongs to.
	// Together they keep the reasoning, the commands, what those commands
	// print, and the answer from running together.
	open        io.Writer
	openSection section
}

// New starts a conversation. A non-empty system prompt stands in front of the
// history. The output writer receives streamed text, the thinking writer the
// streamed reasoning, and the notices writer the command lines and what they
// print.
func New(
	client *llm.Client,
	runner shell.Runner,
	systemPrompt string,
	maxTurns int,
	out, thinking, notices io.Writer,
) *Agent {
	if maxTurns <= 0 {
		maxTurns = 1
	}
	system := strings.TrimSpace(systemPrompt)
	agent := &Agent{
		Client:       client,
		Runner:       runner,
		SystemPrompt: system,
		MaxTurns:     maxTurns,
		Out:          out,
		Thinking:     thinking,
		Notices:      notices,
	}
	if system != "" {
		agent.history = []llm.Message{{Role: llm.RoleSystem, Content: system}}
	}
	return agent
}

// History returns the conversation with the system prompt removed, the form
// the transcript stores.
func (a *Agent) History() []llm.Message {
	if len(a.history) == 0 {
		return nil
	}
	out := make([]llm.Message, 0, len(a.history))
	for _, message := range a.history {
		if message.Role == llm.RoleSystem {
			continue
		}
		out = append(out, message)
	}
	return out
}

// Restore seeds the conversation from a saved transcript, keeping the current
// system prompt when one stands in front.
func (a *Agent) Restore(messages []llm.Message) {
	a.history = make([]llm.Message, 0, len(messages)+1)
	if a.SystemPrompt != "" {
		a.history = append(a.history, llm.Message{Role: llm.RoleSystem, Content: a.SystemPrompt})
	}
	a.history = append(a.history, messages...)
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
	a.history = append(a.history, llm.Message{Role: llm.RoleUser, Content: input})
	a.writeErr = nil
	a.open, a.openSection = nil, sectionNone
	defer a.endLine()
	tools := []llm.Tool{a.tool()}

	for range a.MaxTurns {
		reply, usage, err := a.Client.Complete(ctx, a.history, tools, llm.Stream{
			Text:     a.write,
			Thinking: a.writeThinking,
		})
		if err != nil {
			return "", err
		}
		if a.writeErr != nil {
			return "", a.writeErr
		}
		a.history = append(a.history, reply)
		a.recordUsage(usage)
		if len(reply.ToolCalls) == 0 {
			return reply.Content, nil
		}
		a.runToolCalls(ctx, reply.ToolCalls)
		a.endLine()
	}
	return "", fmt.Errorf("agent: gave up after %d turns without a final answer", a.MaxTurns)
}

// recordUsage stores the context size the endpoint reported, or an estimate
// from the text when it reported none.
func (a *Agent) recordUsage(usage llm.Usage) {
	if total := usage.Total(); total > 0 {
		a.tokens = total
		return
	}
	a.tokens = estimateTokens(a.history)
}

// charsPerToken is the rule of thumb for an endpoint that reports no usage.
const charsPerToken = 4

// estimateTokens applies the rule of thumb of one token per four characters.
func estimateTokens(messages []llm.Message) int {
	chars := 0
	for _, message := range messages {
		chars += len(message.Content) + len(message.ReasoningContent)
		for _, call := range message.ToolCalls {
			chars += len(call.Function.Name) + len(call.Function.Arguments)
		}
	}
	return (chars + charsPerToken - 1) / charsPerToken
}

// write sends one streamed fragment of the answer to the output writer.
func (a *Agent) write(text string) {
	a.writeTo(a.Out, sectionAnswer, text)
}

// writeThinking sends one reasoning fragment to the thinking writer.
func (a *Agent) writeThinking(text string) {
	a.writeTo(a.Thinking, sectionReasoning, text)
}

// writeTo opens its section when a different one left a line unfinished, then
// sends text to its writer. It remembers the first failure so Ask can report
// it and the section whose line the text leaves unfinished.
func (a *Agent) writeTo(w io.Writer, where section, text string) {
	if w == nil || text == "" || a.writeErr != nil {
		return
	}
	a.begin(where)
	if _, err := io.WriteString(w, text); err != nil {
		a.writeErr = fmt.Errorf("agent: write streamed text: %w", err)
		return
	}
	if strings.HasSuffix(text, "\n") {
		a.open, a.openSection = nil, sectionNone
		return
	}
	a.open, a.openSection = w, where
}

// begin ends the line a different section left unfinished, so the section
// about to start does not run into it. Fragments of one section join as they
// stream.
func (a *Agent) begin(next section) {
	if a.openSection == next {
		return
	}
	a.endLine()
}

// endLine finishes the line a section left unfinished, if any.
func (a *Agent) endLine() {
	if a.open == nil {
		return
	}
	a.writeTo(a.open, a.openSection, "\n")
}

// runToolCalls executes each requested call and appends its result.
func (a *Agent) runToolCalls(ctx context.Context, calls []llm.ToolCall) {
	for _, call := range calls {
		result, err := a.dispatch(ctx, call)
		if err != nil {
			result = err.Error()
		}
		a.history = append(a.history, llm.Message{
			Role:       llm.RoleTool,
			ToolCallID: call.ID,
			Content:    result,
		})
	}
}

// dispatch runs one tool call and renders its result as text.
func (a *Agent) dispatch(ctx context.Context, call llm.ToolCall) (string, error) {
	if call.Function.Name != shellToolName {
		return "", fmt.Errorf("agent: unknown tool %q", call.Function.Name)
	}
	var args struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("agent: bad arguments for %s: %w", shellToolName, err)
	}
	if strings.TrimSpace(args.Command) == "" {
		return "", fmt.Errorf("agent: %s called without a command", shellToolName)
	}
	a.notice(args.Command)
	res, err := a.Runner.Run(ctx, args.Command)
	if err != nil {
		return "", err
	}
	result := render(res)
	a.output(result)
	return result, nil
}

// notice reports a command before it runs.
func (a *Agent) notice(command string) {
	a.writeTo(a.Notices, sectionCommand, "$ "+command+"\n")
}

// output shows what a command printed, so the run shows the work and not only
// the request.
func (a *Agent) output(text string) {
	a.writeTo(a.Notices, sectionOutput, text)
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

// tool advertises the shell tool.
func (a *Agent) tool() llm.Tool {
	description := "Run a command line through the system shell and return its combined output and exit code. " + a.Runner.SystemPrompt()
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name:        shellToolName,
			Description: description,
			Parameters:  json.RawMessage(toolParameters),
		},
	}
}

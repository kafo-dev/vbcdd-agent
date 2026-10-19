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

// Agent holds one conversation and the loop that advances it.
type Agent struct {
	Client       *llm.Client
	Runner       shell.Runner
	SystemPrompt string
	MaxTurns     int
	// Out receives the assistant's text as the model streams it.
	Out     io.Writer
	Notices io.Writer

	history  []llm.Message
	tokens   int
	writeErr error
}

// New starts a conversation. A non-empty system prompt stands in front of the
// history. The output writer receives streamed text; the notices writer
// receives the command lines.
func New(
	client *llm.Client,
	runner shell.Runner,
	systemPrompt string,
	maxTurns int,
	out, notices io.Writer,
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
	tools := []llm.Tool{a.tool()}

	for range a.MaxTurns {
		reply, usage, err := a.Client.Complete(ctx, a.history, tools, a.write)
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
		chars += len(message.Content)
		for _, call := range message.ToolCalls {
			chars += len(call.Function.Name) + len(call.Function.Arguments)
		}
	}
	return (chars + charsPerToken - 1) / charsPerToken
}

// write sends one streamed fragment to the output writer, remembering the
// first failure so Ask can report it.
func (a *Agent) write(text string) {
	if a.Out == nil || a.writeErr != nil {
		return
	}
	if _, err := io.WriteString(a.Out, text); err != nil {
		a.writeErr = fmt.Errorf("agent: write streamed text: %w", err)
	}
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
	return render(res), nil
}

// notice reports a command before it runs.
func (a *Agent) notice(command string) {
	if a.Notices == nil {
		return
	}
	fmt.Fprintf(a.Notices, "$ %s\n", command)
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

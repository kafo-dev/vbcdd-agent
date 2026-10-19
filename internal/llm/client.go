// Package llm speaks the OpenAI chat-completions protocol with tool calls.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Role names the author of a message.
type Role string

// The roles the protocol defines.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is one entry of the conversation.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
	// ReasoningContent holds the chain of thought a reasoning model emitted
	// before its answer. A request that carries tools must send it back, or
	// the model loses the reasoning it already did.
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// ToolCall is one function the model asked the program to run.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall names the tool and carries its arguments as JSON text.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool advertises one function to the model.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function half of a tool definition.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// Usage counts the tokens one request consumed.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Total reports the tokens the request used, summing the halves when the
// endpoint omits the total.
func (u Usage) Total() int {
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.PromptTokens + u.CompletionTokens
}

type chatRequest struct {
	Model           string         `json:"model"`
	Messages        []Message      `json:"messages"`
	Tools           []Tool         `json:"tools,omitempty"`
	Stream          bool           `json:"stream,omitempty"`
	StreamOptions   *streamOptions `json:"stream_options,omitempty"`
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
}

// streamOptions asks the endpoint to report token usage in the final event.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

const (
	// requestTimeout bounds one HTTP round trip.
	requestTimeout = 10 * time.Minute
	// maxErrorBody caps how much of an error response the message quotes.
	maxErrorBody = 4 << 10
	// scannerInit is the starting buffer for one stream event.
	scannerInit = 64 << 10
	// scannerMax bounds one stream event.
	scannerMax = 1 << 20
)

// Client calls one OpenAI-compatible endpoint.
type Client struct {
	BaseURL string
	APIKey  string
	Model   string
	// ReasoningEffort asks the endpoint to spend that much on reasoning, as
	// `low`, `high`, or `max`. An empty value leaves the choice to the
	// endpoint.
	ReasoningEffort string
	// Headers carries extra request headers, such as a provider's
	// authentication scheme. A custom header wins over the default the
	// client would set.
	Headers map[string]string
	HTTP    *http.Client
}

// New builds a client bound to an endpoint. An empty HTTP client gets the
// default.
func New(baseURL, apiKey, model string, headers map[string]string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: requestTimeout}
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Model:   model,
		Headers: headers,
		HTTP:    httpClient,
	}
}

// streamDelta is one incremental piece of the assistant's reply. A reasoning
// model streams its chain of thought through ReasoningContent before it
// streams the answer through Content.
type streamDelta struct {
	Content          string                   `json:"content"`
	ReasoningContent string                   `json:"reasoning_content"`
	ToolCalls        []streamToolCallFragment `json:"tool_calls"`
}

// streamToolCallFragment is one fragment of a tool call. The protocol splits a
// call across several fragments: the first carries the id and the name, and the
// arguments arrive in pieces that the reader joins.
type streamToolCallFragment struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// streamChunk is one server-sent event of the reply.
type streamChunk struct {
	Choices []struct {
		Delta        streamDelta `json:"delta"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage    `json:"usage,omitempty"`
	Error *apiError `json:"error,omitempty"`
}

// Stream carries the callbacks that receive a reply as it arrives. A nil
// callback drops its fragments.
type Stream struct {
	// Text receives the answer's content fragments.
	Text func(string)
	// Thinking receives the reasoning fragments that precede the answer.
	Thinking func(string)
}

// Complete sends the conversation and returns the assistant's reply and the
// tokens it consumed. It streams the reply, calling the stream callbacks with
// each fragment as it arrives.
func (c *Client) Complete(
	ctx context.Context,
	messages []Message,
	tools []Tool,
	stream Stream,
) (Message, Usage, error) {
	if c.BaseURL == "" {
		return Message{}, Usage{}, fmt.Errorf("llm: base url is empty")
	}
	if c.Model == "" {
		return Message{}, Usage{}, fmt.Errorf("llm: model is empty")
	}
	if len(messages) == 0 {
		return Message{}, Usage{}, fmt.Errorf("llm: no messages to send")
	}
	request := chatRequest{
		Model:           c.Model,
		Messages:        messages,
		Tools:           tools,
		Stream:          true,
		StreamOptions:   &streamOptions{IncludeUsage: true},
		ReasoningEffort: c.ReasoningEffort,
	}
	body, err := json.Marshal(request)
	if err != nil {
		return Message{}, Usage{}, fmt.Errorf("llm: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, Usage{}, fmt.Errorf("llm: build request: %w", err)
	}
	for name, value := range c.Headers {
		req.Header.Set(name, value)
	}
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if req.Header.Get("Accept") == "" {
		req.Header.Set("Accept", "text/event-stream")
	}
	if c.APIKey != "" && req.Header.Get("Authorization") == "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Message{}, Usage{}, fmt.Errorf("llm: request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, readErr := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		if readErr != nil {
			return Message{}, Usage{}, fmt.Errorf("llm: %s: %w", resp.Status, readErr)
		}
		return Message{}, Usage{}, fmt.Errorf("llm: %s: %s", resp.Status, truncate(string(raw), maxErrorBody))
	}
	return readStream(resp.Body, stream)
}

// readStream parses the server-sent events into one assistant message and the
// tokens the request reported.
func readStream(body io.Reader, stream Stream) (Message, Usage, error) {
	state := streamState{message: Message{Role: RoleAssistant}, stream: stream}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, scannerInit), scannerMax)
	for scanner.Scan() {
		payload, ok := eventPayload(scanner.Text())
		if !ok {
			continue
		}
		if payload == streamDone {
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return Message{}, Usage{}, fmt.Errorf("llm: decode stream: %w", err)
		}
		if chunk.Error != nil {
			return Message{}, Usage{}, fmt.Errorf("llm: %s: %s", chunk.Error.Type, chunk.Error.Message)
		}
		state.apply(chunk)
	}
	if err := scanner.Err(); err != nil {
		return Message{}, Usage{}, fmt.Errorf("llm: read stream: %w", err)
	}
	state.message.ToolCalls = state.calls
	return state.message, state.usage, nil
}

// streamState accumulates one reply across its events.
type streamState struct {
	message Message
	calls   []ToolCall
	usage   Usage
	stream  Stream
}

// apply folds one event into the reply, reporting new text to the callbacks.
func (s *streamState) apply(chunk streamChunk) {
	if chunk.Usage != nil {
		s.usage = *chunk.Usage
	}
	for _, choice := range chunk.Choices {
		if choice.Delta.ReasoningContent != "" {
			s.message.ReasoningContent += choice.Delta.ReasoningContent
			if s.stream.Thinking != nil {
				s.stream.Thinking(choice.Delta.ReasoningContent)
			}
		}
		if choice.Delta.Content != "" {
			s.message.Content += choice.Delta.Content
			if s.stream.Text != nil {
				s.stream.Text(choice.Delta.Content)
			}
		}
		for _, fragment := range choice.Delta.ToolCalls {
			s.calls = mergeToolCall(s.calls, fragment)
		}
	}
}

// eventPayload returns the data of one event line, or false when the line
// carries no payload.
func eventPayload(line string) (string, bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, ":") {
		return "", false
	}
	payload, ok := strings.CutPrefix(line, "data:")
	if !ok {
		return "", false
	}
	return strings.TrimSpace(payload), true
}

// mergeToolCall joins one fragment into the call at its index, growing the
// slice until the index lands inside it.
func mergeToolCall(calls []ToolCall, fragment streamToolCallFragment) []ToolCall {
	for len(calls) <= fragment.Index {
		calls = append(calls, ToolCall{})
	}
	call := &calls[fragment.Index]
	if fragment.ID != "" {
		call.ID = fragment.ID
	}
	if fragment.Type != "" {
		call.Type = fragment.Type
	}
	if fragment.Function.Name != "" {
		call.Function.Name = fragment.Function.Name
	}
	call.Function.Arguments += fragment.Function.Arguments
	return calls
}

// truncate cuts text to at most limit bytes.
func truncate(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}

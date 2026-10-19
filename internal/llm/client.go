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
	Role       Role       `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
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

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream,omitempty"`
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

// streamDelta is one incremental piece of the assistant's reply.
type streamDelta struct {
	Content   string                   `json:"content"`
	ToolCalls []streamToolCallFragment `json:"tool_calls"`
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
	Error *apiError `json:"error,omitempty"`
}

// Complete sends the conversation and returns the assistant's reply. It streams
// the reply, calling onText with each content fragment as it arrives. A nil
// onText discards the fragments but still assembles the message.
func (c *Client) Complete(ctx context.Context, messages []Message, tools []Tool, onText func(string)) (Message, error) {
	if c.BaseURL == "" {
		return Message{}, fmt.Errorf("llm: base url is empty")
	}
	if c.Model == "" {
		return Message{}, fmt.Errorf("llm: model is empty")
	}
	if len(messages) == 0 {
		return Message{}, fmt.Errorf("llm: no messages to send")
	}
	request := chatRequest{Model: c.Model, Messages: messages, Tools: tools, Stream: true}
	body, err := json.Marshal(request)
	if err != nil {
		return Message{}, fmt.Errorf("llm: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, fmt.Errorf("llm: build request: %w", err)
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
		return Message{}, fmt.Errorf("llm: request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, readErr := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		if readErr != nil {
			return Message{}, fmt.Errorf("llm: %s: %w", resp.Status, readErr)
		}
		return Message{}, fmt.Errorf("llm: %s: %s", resp.Status, truncate(string(raw), maxErrorBody))
	}
	return readStream(resp.Body, onText)
}

// readStream parses the server-sent events into one assistant message.
func readStream(body io.Reader, onText func(string)) (Message, error) {
	state := streamState{message: Message{Role: RoleAssistant}, onText: onText}
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
			return Message{}, fmt.Errorf("llm: decode stream: %w", err)
		}
		if chunk.Error != nil {
			return Message{}, fmt.Errorf("llm: %s: %s", chunk.Error.Type, chunk.Error.Message)
		}
		state.apply(chunk)
	}
	if err := scanner.Err(); err != nil {
		return Message{}, fmt.Errorf("llm: read stream: %w", err)
	}
	state.message.ToolCalls = state.calls
	return state.message, nil
}

// streamState accumulates one reply across its events.
type streamState struct {
	message Message
	calls   []ToolCall
	onText  func(string)
}

// apply folds one event into the reply, reporting new text to the callback.
func (s *streamState) apply(chunk streamChunk) {
	for _, choice := range chunk.Choices {
		if choice.Delta.Content != "" {
			s.message.Content += choice.Delta.Content
			if s.onText != nil {
				s.onText(choice.Delta.Content)
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

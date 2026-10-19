// Package llm speaks the OpenAI chat-completions protocol with tool calls.
package llm

import (
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
}

type chatResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Error *apiError `json:"error,omitempty"`
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
	// maxResponseBody caps how much of a successful response the client reads.
	maxResponseBody = 1 << 20
)

// Client calls one OpenAI-compatible endpoint.
type Client struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

// New builds a client bound to an endpoint. An empty HTTP client gets the
// default.
func New(baseURL, apiKey, model string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: requestTimeout}
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Model:   model,
		HTTP:    httpClient,
	}
}

// Complete sends the conversation and returns the assistant's reply.
func (c *Client) Complete(ctx context.Context, messages []Message, tools []Tool) (Message, error) {
	if c.BaseURL == "" {
		return Message{}, fmt.Errorf("llm: base url is empty")
	}
	if c.Model == "" {
		return Message{}, fmt.Errorf("llm: model is empty")
	}
	if len(messages) == 0 {
		return Message{}, fmt.Errorf("llm: no messages to send")
	}
	body, err := json.Marshal(chatRequest{Model: c.Model, Messages: messages, Tools: tools})
	if err != nil {
		return Message{}, fmt.Errorf("llm: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, fmt.Errorf("llm: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("llm: request: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBody))
	if err != nil {
		return Message{}, fmt.Errorf("llm: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return Message{}, fmt.Errorf("llm: %s: %s", resp.Status, truncate(string(raw), maxErrorBody))
	}
	var parsed chatResponse
	if err = json.Unmarshal(raw, &parsed); err != nil {
		return Message{}, fmt.Errorf("llm: decode response: %w", err)
	}
	if parsed.Error != nil {
		return Message{}, fmt.Errorf("llm: %s: %s", parsed.Error.Type, parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return Message{}, fmt.Errorf("llm: response carries no choices")
	}
	reply := parsed.Choices[0].Message
	reply.Role = RoleAssistant
	return reply, nil
}

// truncate cuts text to at most limit bytes.
func truncate(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}

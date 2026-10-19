// Package transcript saves and reloads conversations as JSON Lines.
//
// One file holds one conversation. Each line is a message, so a reader keeps
// what it can understand even when a later line is corrupt. The system prompt
// stays out: a resumed conversation takes the prompt the run supplies.
package transcript

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"vbcdd.kafo.dev/agent/internal/llm"
)

const (
	fileSuffix = ".jsonl"
	dirMode    = 0o750
	fileMode   = 0o600
	timeFormat = "20060102T150405"
	// scannerInit is the starting buffer for a transcript line.
	scannerInit = 64 << 10
	// scannerMax bounds one transcript line.
	scannerMax = 1 << 20
)

// Session is one saved conversation.
type Session struct {
	// ID names the file without its extension.
	ID      string
	Path    string
	Started time.Time
}

// Store owns the directory holding conversations.
type Store struct {
	Dir string
}

// Create starts a conversation and opens its file for appending.
func (s Store) Create() (*Writer, error) {
	if s.Dir == "" {
		return nil, fmt.Errorf("transcript: no directory")
	}
	if err := os.MkdirAll(s.Dir, dirMode); err != nil {
		return nil, fmt.Errorf("transcript: create directory: %w", err)
	}
	started := time.Now()
	id := started.Format(timeFormat)
	path := filepath.Join(s.Dir, id+fileSuffix)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, fileMode)
	if err != nil {
		return nil, fmt.Errorf("transcript: open %s: %w", path, err)
	}
	return &Writer{Session: Session{ID: id, Path: path, Started: started}, file: file}, nil
}

// Append opens an existing conversation for appending.
func (s Store) Append(session Session) (*Writer, error) {
	if session.ID == "" {
		return nil, fmt.Errorf("transcript: session has no id")
	}
	if session.Path == "" {
		return nil, fmt.Errorf("transcript: session has no path")
	}
	file, err := os.OpenFile(session.Path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, fileMode)
	if err != nil {
		return nil, fmt.Errorf("transcript: open %s: %w", session.Path, err)
	}
	return &Writer{Session: session, file: file}, nil
}

// List returns every saved conversation, newest first.
func (s Store) List() ([]Session, error) {
	entries, err := os.ReadDir(s.Dir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("transcript: list %s: %w", s.Dir, err)
	}
	sessions := make([]Session, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), fileSuffix) {
			continue
		}
		id := strings.TrimSuffix(entry.Name(), fileSuffix)
		started, parseErr := time.ParseInLocation(timeFormat, id, time.Local)
		if parseErr != nil {
			started = time.Time{}
		}
		sessions = append(sessions, Session{
			ID:      id,
			Path:    filepath.Join(s.Dir, entry.Name()),
			Started: started,
		})
	}
	sort.Slice(sessions, func(i, j int) bool { return sessions[i].ID > sessions[j].ID })
	return sessions, nil
}

// Latest returns the newest conversation. It reports a missing directory as
// an empty result rather than an error.
func (s Store) Latest() (Session, bool, error) {
	sessions, err := s.List()
	if err != nil {
		return Session{}, false, err
	}
	if len(sessions) == 0 {
		return Session{}, false, nil
	}
	return sessions[0], true, nil
}

// Open returns a conversation by ID together with its messages.
func (s Store) Open(id string) (Session, []llm.Message, error) {
	if s.Dir == "" {
		return Session{}, nil, fmt.Errorf("transcript: no directory")
	}
	if id == "" {
		return Session{}, nil, fmt.Errorf("transcript: no session id")
	}
	path := filepath.Join(s.Dir, id+fileSuffix)
	messages, err := Read(path)
	if err != nil {
		return Session{}, nil, err
	}
	started, parseErr := time.ParseInLocation(timeFormat, id, time.Local)
	if parseErr != nil {
		started = time.Time{}
	}
	return Session{ID: id, Path: path, Started: started}, messages, nil
}

// Read loads the messages from a transcript file.
func Read(path string) ([]llm.Message, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("transcript: open %s: %w", path, err)
	}
	defer file.Close()

	messages := []llm.Message{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, scannerInit), scannerMax)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var message llm.Message
		if err = json.Unmarshal([]byte(line), &message); err != nil {
			return nil, fmt.Errorf("transcript: %s: %w", path, err)
		}
		if message.Role == llm.RoleSystem {
			continue
		}
		messages = append(messages, message)
	}
	if err = scanner.Err(); err != nil {
		return nil, fmt.Errorf("transcript: read %s: %w", path, err)
	}
	return messages, nil
}

// Writer appends messages to one conversation.
type Writer struct {
	Session Session
	file    *os.File
}

// Append writes one message as a line.
func (w *Writer) Append(message llm.Message) error {
	if w.file == nil {
		return fmt.Errorf("transcript: writer is closed")
	}
	if message.Role == llm.RoleSystem {
		return nil
	}
	raw, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("transcript: encode message: %w", err)
	}
	if _, err = w.file.Write(append(raw, '\n')); err != nil {
		return fmt.Errorf("transcript: append %s: %w", w.Session.Path, err)
	}
	return nil
}

// Close releases the file.
func (w *Writer) Close() error {
	if w.file == nil {
		return nil
	}
	err := w.file.Close()
	w.file = nil
	if err != nil {
		return fmt.Errorf("transcript: close %s: %w", w.Session.Path, err)
	}
	return nil
}

# vbcdd-agent

A terminal agent that answers by running shell commands. It talks to an OpenAI-compatible chat-completions endpoint, advertises one tool, and saves every conversation as JSON Lines.

> **Warning: Large Language Model-Generated Code**
>
> This code functions as of the time of writing, yet it has received little to no human review. It must not be regarded as reliable: bugs and stability issues are to be expected.

```text
agent(0)> list the Go files changed in the last commit
$ git show --name-only --pretty=format: HEAD
cmd/vbcdd-agent/main.go
internal/agent/agent.go
[exit code 0]
The last commit touched the entry point and the tool loop.
```

## Build

```sh
go build ./...
```

## Run

```sh
vbcdd-agent [flags] [message]
```

A message argument answers one request and exits. Standard input supplies the message when no argument names one, so a pipeline works:

```sh
vbcdd-agent "how much free memory does this machine have?" < /dev/null
```

An interactive prompt starts when standard input refers to a terminal and no argument names a message. `quit`, `exit`, and Control-D leave it. The prompt prints the size of the context, in tokens, that the endpoint reported for the last request; when it reported none, the count falls back to one token per four characters.

| Flag | Effect |
| --- | --- |
| `-session` | Resume the conversation with this id |
| `-continue` | Resume the most recent conversation |
| `-list` | Print saved conversations and exit |
| `-once` | Answer one message from standard input, then exit |

Every command runs through `$SHELL` with `-c`, in the current working directory.

## Configuration

The configuration file lives at `$XDG_CONFIG_HOME/vbcdd/agent/config.toml`.

```toml
# An endpoint speaking the OpenAI chat-completions protocol.
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
model = "gpt-4o-mini"

# How much the model may spend on reasoning before it answers: `low`, `high`,
# or `max`, as the endpoint defines them. An empty value leaves the choice to
# the endpoint. The reasoning streams to standard error as it arrives.
reasoning_effort = "low"

# A command that prints the key. It runs through `$SHELL -c` on every start and
# wins over `api_key`, which suits a secret manager.
api_key_command = "op read op://Private/OpenAI/credential"

# The standing instruction. The program ships none, so this key is the only
# place a prompt enters the conversation. `system_prompt_file` loads it from a
# file instead.
system_prompt = "Answer in one paragraph."

# Bounds on a single request.
max_turns = 20
shell_timeout_seconds = 60
max_output_bytes = 65536

# Extra request headers, winning over the header the client would set.
[headers]
"X-Api-Key" = "..."
```

A header in `[headers]` overrides the `Authorization`, `Content-Type`, and `Accept` header the client would set. A provider that authenticates under its own name therefore works with its name in `[headers]` and an empty `api_key`.

A reasoning model streams its chain of thought before its answer. The program writes those fragments to standard error as they arrive, keeping standard output for the answer alone. Each command and the output it prints join the reasoning there, each on its own line. It keeps that reasoning in the conversation and sends it back on later turns, as the protocol requires once tools are in play.

Each environment variable overrides the file:

| Variable | Effect |
| --- | --- |
| `VBCDD_AGENT_BASE_URL` | Endpoint |
| `VBCDD_AGENT_API_KEY` | Key |
| `VBCDD_AGENT_MODEL` | Model name |
| `VBCDD_AGENT_CONFIG_DIR` | Configuration directory |
| `VBCDD_AGENT_DATA_DIR` | Data directory |
| `VBCDD_AGENT_LOG_LEVEL` | `debug`, `info`, `warn`, or `error` |
| `VBCDD_AGENT_LOG_FILE` | Log file, or `stderr` |

## Conversations

Each conversation is one file below `$XDG_DATA_HOME/vbcdd/agent/sessions/`. Every line holds one message, so a reader recovers the lines it understands when a later line is corrupt. The system prompt stays out of the file; a resumed conversation takes the prompt the run supplies.

## License

This is free and unencumbered software released into the public domain. See `LICENSE` for the full text.

use super::*;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

struct FakeTransport {
    responses: Mutex<VecDeque<Value>>,
    requests: Mutex<Vec<Value>>,
}

impl FakeTransport {
    fn new(responses: Vec<Value>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

impl ResponseTransport for FakeTransport {
    fn create_response(&self, request: &Value) -> AgentResult<Value> {
        self.requests.lock().unwrap().push(request.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| Box::new(AgentError("no fake response queued".to_string())) as _)
    }
}

#[test]
fn agent_returns_message_without_tool_calls() {
    let transport = FakeTransport::new(vec![json!({
        "id": "resp_1",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "hello from model"}]
        }]
    })]);
    let agent = Agent::new(
        transport,
        ToolExecutor::new(test_temp_dir("no-tools")),
        "gpt-5.4".to_string(),
        DEFAULT_INSTRUCTIONS.to_string(),
    );

    let result = agent.run("say hi").unwrap();

    assert_eq!(result, "hello from model");
}

#[test]
fn agent_executes_tool_calls_and_feeds_outputs_back() {
    let temp_dir = test_temp_dir("tool-round-trip");
    let transport = FakeTransport::new(vec![
        json!({
            "id": "resp_1",
            "output": [{
                "type": "function_call",
                "name": "write_file",
                "call_id": "call_1",
                "arguments": "{\"path\":\"artifact.txt\",\"content\":\"created by test\"}"
            }]
        }),
        json!({
            "id": "resp_2",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}]
            }]
        }),
    ]);

    let agent = Agent::new(
        transport,
        ToolExecutor::new(temp_dir.clone()),
        "gpt-5.4".to_string(),
        DEFAULT_INSTRUCTIONS.to_string(),
    );

    let result = agent.run("create a file").unwrap();

    assert_eq!(result, "done");
    assert_eq!(
        fs::read_to_string(temp_dir.join("artifact.txt")).unwrap(),
        "created by test"
    );

    let requests = agent.transport.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["model"], "gpt-5.4");
    assert_eq!(
        requests[0]["input"][0]["content"][0]["text"],
        "create a file"
    );
    assert_eq!(requests[1]["previous_response_id"], "resp_1");
    assert_eq!(requests[1]["input"][0]["type"], "function_call_output");
    assert_eq!(requests[1]["input"][0]["call_id"], "call_1");
    assert!(
        requests[1]["input"][0]["output"]
            .as_str()
            .unwrap()
            .contains("Wrote")
    );
}

#[test]
fn agent_surfaces_unknown_tool_errors_to_the_model() {
    let transport = FakeTransport::new(vec![
        json!({
            "id": "resp_1",
            "output": [{
                "type": "function_call",
                "name": "does_not_exist",
                "call_id": "call_1",
                "arguments": "{}"
            }]
        }),
        json!({
            "id": "resp_2",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "handled"}]
            }]
        }),
    ]);

    let agent = Agent::new(
        transport,
        ToolExecutor::new(test_temp_dir("unknown-tool")),
        "gpt-5.4".to_string(),
        DEFAULT_INSTRUCTIONS.to_string(),
    );

    let result = agent.run("try an invalid tool").unwrap();

    assert_eq!(result, "handled");
    let requests = agent.transport.requests();
    assert!(
        requests[1]["input"][0]["output"]
            .as_str()
            .unwrap()
            .contains("Unknown tool")
    );
}

#[test]
fn agent_rejects_invalid_tool_arguments() {
    let transport = FakeTransport::new(vec![json!({
        "id": "resp_1",
        "output": [{
            "type": "function_call",
            "name": "write_file",
            "call_id": "call_1",
            "arguments": "{not-json}"
        }]
    })]);
    let agent = Agent::new(
        transport,
        ToolExecutor::new(test_temp_dir("bad-args")),
        "gpt-5.4".to_string(),
        DEFAULT_INSTRUCTIONS.to_string(),
    );

    let error = agent.run("bad args").unwrap_err().to_string();

    assert!(error.contains("invalid JSON arguments"));
}

#[test]
fn agent_stops_when_tool_calls_repeat_without_progress() {
    let transport = FakeTransport::new(vec![
        json!({
            "id": "resp_1",
            "output": [{
                "type": "function_call",
                "name": "read_file",
                "call_id": "call_1",
                "arguments": "{\"path\":\"missing.txt\"}"
            }]
        }),
        json!({
            "id": "resp_2",
            "output": [{
                "type": "function_call",
                "name": "read_file",
                "call_id": "call_2",
                "arguments": "{\"path\":\"missing.txt\"}"
            }]
        }),
    ]);
    let agent = Agent::new(
        transport,
        ToolExecutor::new(test_temp_dir("semantic-stop")),
        "gpt-5.4".to_string(),
        DEFAULT_INSTRUCTIONS.to_string(),
    );

    let error = agent.run("loop forever").unwrap_err().to_string();

    assert!(error.contains("semantic stop condition"));
    assert_eq!(agent.transport.requests().len(), 2);
}

#[test]
fn tool_executor_reads_and_writes_files() {
    let temp_dir = test_temp_dir("files");
    let tools = ToolExecutor::new(temp_dir.clone());

    let write_result = tools
        .execute(
            "write_file",
            &json!({"path": "nested/note.txt", "content": "hello"}),
        )
        .unwrap();
    let read_result = tools
        .execute("read_file", &json!({"path": "nested/note.txt"}))
        .unwrap();

    assert!(write_result.contains("Wrote"));
    assert_eq!(read_result, "hello");
    assert_eq!(
        fs::read_to_string(temp_dir.join("nested").join("note.txt")).unwrap(),
        "hello"
    );
}

#[test]
fn tool_executor_runs_shell_commands() {
    let tools = ToolExecutor::new(test_temp_dir("shell"));
    let command = if cfg!(windows) {
        "Write-Output hello-shell"
    } else {
        "printf hello-shell"
    };

    let output = tools
        .execute("execute_shell", &json!({ "command": command }))
        .unwrap();

    assert!(output.contains("hello-shell"));
}

#[test]
fn collect_output_text_joins_multiple_chunks() {
    let text = collect_output_text(&json!({
        "output": [{
            "type": "message",
            "content": [
                {"type": "output_text", "text": "first"},
                {"type": "output_text", "text": "second"}
            ]
        }]
    }));

    assert_eq!(text, "first\nsecond");
}

#[test]
fn parse_cli_rejects_missing_task() {
    let error = parse_cli(vec!["gpt-4.1-mini".to_string()])
        .unwrap_err()
        .to_string();

    assert!(error.contains("missing required positional argument: task"));
}

#[test]
fn parse_cli_uses_remaining_arguments_as_task() {
    let cli = parse_cli(vec![
        "gpt-4.1-mini".to_string(),
        "say".to_string(),
        "hi".to_string(),
    ])
    .unwrap();

    assert_eq!(
        cli,
        CliArgs {
            task: "say hi".to_string(),
            model: "gpt-4.1-mini".to_string(),
        }
    );
}

#[test]
fn parse_cli_rejects_missing_model() {
    let error = parse_cli(Vec::<String>::new()).unwrap_err().to_string();

    assert!(error.contains("missing required positional argument: model"));
}

fn test_temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "ouro-agent-test-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

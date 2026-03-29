use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_INSTRUCTIONS: &str = "You are a helpful assistant. Be concise.";

type AgentResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CliArgs {
    pub(crate) task: String,
    pub(crate) model: String,
}

fn main() -> AgentResult<()> {
    let cli = parse_cli(env::args().skip(1))?;

    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| AgentError("OPENAI_API_KEY is not set".to_string()))?;
    let base_url = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());

    let transport = HttpTransport::new(base_url, api_key);
    let agent = Agent::new(
        transport,
        ToolExecutor::new(env::current_dir()?),
        cli.model,
        DEFAULT_INSTRUCTIONS.to_string(),
    );

    println!("{}", agent.run(&cli.task)?);
    Ok(())
}

#[derive(Debug)]
struct AgentError(String);

impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for AgentError {}

pub(crate) fn parse_cli<I>(args: I) -> AgentResult<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let args: Vec<String> = args.into_iter().collect();

    if args.is_empty() {
        return Err(Box::new(AgentError(
            "missing required positional argument: model".to_string(),
        )));
    }
    if args.len() == 1 {
        return Err(Box::new(AgentError(
            "missing required positional argument: task".to_string(),
        )));
    }

    let model = args[0].clone();
    let task = args[1..].join(" ");

    Ok(CliArgs { task, model })
}

trait ResponseTransport {
    fn create_response(&self, request: &Value) -> AgentResult<Value>;
}

struct HttpTransport {
    client: ureq::Agent,
    base_url: String,
    api_key: String,
}

impl HttpTransport {
    fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: ureq::AgentBuilder::new().build(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }
}

impl ResponseTransport for HttpTransport {
    fn create_response(&self, request: &Value) -> AgentResult<Value> {
        let response = self
            .client
            .post(&format!("{}/responses", self.base_url))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(request.clone())?;

        Ok(response.into_json()?)
    }
}

struct Agent<T: ResponseTransport> {
    transport: T,
    tools: ToolExecutor,
    model: String,
    instructions: String,
}

impl<T: ResponseTransport> Agent<T> {
    fn new(transport: T, tools: ToolExecutor, model: String, instructions: String) -> Self {
        Self {
            transport,
            tools,
            model,
            instructions,
        }
    }

    fn run(&self, user_message: &str) -> AgentResult<String> {
        let mut follow_up_input = json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": user_message}],
        }]);
        let mut previous_response_id = None;
        let mut last_step = None;

        loop {
            let request = self.build_request(previous_response_id.as_deref(), follow_up_input);
            let response = self.transport.create_response(&request)?;
            let tool_calls = extract_tool_calls(&response)?;

            if tool_calls.is_empty() {
                let text = collect_output_text(&response);
                if text.is_empty() {
                    return Err(Box::new(AgentError(
                        "response did not contain tool calls or message text".to_string(),
                    )));
                }
                return Ok(text);
            }

            previous_response_id = Some(response_id(&response)?);
            let mut outputs = Vec::with_capacity(tool_calls.len());
            let mut current_step = StepFingerprint::with_capacity(tool_calls.len());
            for call in tool_calls {
                let result = match self.tools.execute(&call.name, &call.arguments) {
                    Ok(output) => output,
                    Err(error) => format!("Error: {error}"),
                };
                current_step.push(&call, &result);
                outputs.push(json!({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }));
            }

            if last_step.as_ref() == Some(&current_step) {
                return Err(Box::new(AgentError(
                    "semantic stop condition triggered: repeated identical tool calls produced identical outputs".to_string(),
                )));
            }

            last_step = Some(current_step);
            follow_up_input = Value::Array(outputs);
        }
    }

    fn build_request(&self, previous_response_id: Option<&str>, input: Value) -> Value {
        let mut request = json!({
            "model": self.model,
            "instructions": self.instructions,
            "input": input,
            "tools": tool_definitions(),
        });

        if let Some(previous_response_id) = previous_response_id {
            request["previous_response_id"] = json!(previous_response_id);
        }

        request
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolCall {
    name: String,
    call_id: String,
    arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolCallOutcome {
    name: String,
    arguments: Value,
    output: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StepFingerprint {
    outcomes: Vec<ToolCallOutcome>,
}

impl StepFingerprint {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            outcomes: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, call: &ToolCall, output: &str) {
        self.outcomes.push(ToolCallOutcome {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            output: output.to_string(),
        });
    }
}

fn extract_tool_calls(response: &Value) -> AgentResult<Vec<ToolCall>> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| AgentError("response.output was not an array".to_string()))?;

    let mut tool_calls = Vec::new();
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            continue;
        }

        let name = item
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError("function call missing name".to_string()))?;
        let call_id = item
            .get("call_id")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError("function call missing call_id".to_string()))?;
        let arguments = item
            .get("arguments")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError("function call missing arguments".to_string()))?;
        let arguments = serde_json::from_str(arguments).map_err(|error| {
            AgentError(format!("invalid JSON arguments for tool '{name}': {error}"))
        })?;

        tool_calls.push(ToolCall {
            name: name.to_string(),
            call_id: call_id.to_string(),
            arguments,
        });
    }

    Ok(tool_calls)
}

fn collect_output_text(response: &Value) -> String {
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return String::new();
    };

    let mut chunks = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for part in content {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            chunks.push(text.to_string());
                        }
                    }
                }
            }
            Some("output_text") => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    chunks.push(text.to_string());
                }
            }
            _ => {}
        }
    }

    chunks.join("\n")
}

fn response_id(response: &Value) -> AgentResult<String> {
    response
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| Box::new(AgentError("response missing id".to_string())) as _)
}

fn tool_definitions() -> Value {
    json!([
        {
            "type": "function",
            "name": "execute_shell",
            "description": "Execute a shell command on the local machine and return combined output.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }
        },
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a UTF-8 text file from disk.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute file path."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        },
        {
            "type": "function",
            "name": "write_file",
            "description": "Write UTF-8 text content to disk, creating parent directories when needed.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute file path."
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write."
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }
        }
    ])
}

struct ToolExecutor {
    working_dir: PathBuf,
}

impl ToolExecutor {
    fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }

    fn execute(&self, name: &str, arguments: &Value) -> AgentResult<String> {
        match name {
            "execute_shell" => self.execute_shell(required_string(arguments, "command")?),
            "read_file" => self.read_file(required_string(arguments, "path")?),
            "write_file" => self.write_file(
                required_string(arguments, "path")?,
                required_string(arguments, "content")?,
            ),
            _ => Err(Box::new(AgentError(format!("Unknown tool '{name}'")))),
        }
    }

    fn execute_shell(&self, command: &str) -> AgentResult<String> {
        let output = platform_shell(&self.working_dir, command).output()?;
        let mut combined = String::new();
        combined.push_str(&String::from_utf8_lossy(&output.stdout));
        combined.push_str(&String::from_utf8_lossy(&output.stderr));

        if combined.is_empty() {
            combined = match output.status.code() {
                Some(code) => format!("Command exited with status {code}"),
                None => "Command exited without a status code".to_string(),
            };
        } else if !output.status.success() {
            match output.status.code() {
                Some(code) => combined.push_str(&format!("\n[exit status: {code}]")),
                None => combined.push_str("\n[exit status: terminated by signal]"),
            }
        }

        Ok(combined)
    }

    fn read_file(&self, path: &str) -> AgentResult<String> {
        Ok(fs::read_to_string(self.resolve_path(path))?)
    }

    fn write_file(&self, path: &str, content: &str) -> AgentResult<String> {
        let path = self.resolve_path(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, content)?;
        Ok(format!(
            "Wrote {} bytes to {}",
            content.len(),
            path.display()
        ))
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.working_dir.join(path)
        }
    }
}

fn required_string<'a>(arguments: &'a Value, field: &str) -> AgentResult<&'a str> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| Box::new(AgentError(format!("missing string field '{field}'"))) as _)
}

#[cfg(windows)]
fn platform_shell(working_dir: &Path, command: &str) -> Command {
    let mut cmd = Command::new("powershell");
    cmd.current_dir(working_dir)
        .arg("-NoProfile")
        .arg("-Command")
        .arg(command);
    cmd
}

#[cfg(not(windows))]
fn platform_shell(working_dir: &Path, command: &str) -> Command {
    let mut cmd = Command::new("sh");
    cmd.current_dir(working_dir).arg("-lc").arg(command);
    cmd
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

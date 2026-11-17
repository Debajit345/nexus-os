use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    UnterminatedQuote,
    UnexpectedToken(String),
    MissingRedirectTarget,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnterminatedQuote => write!(f, "syntax error: unterminated quote"),
            Self::UnexpectedToken(tok) => write!(f, "syntax error near unexpected token `{}`", tok),
            Self::MissingRedirectTarget => write!(f, "syntax error: missing redirect target"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RedirectOut {
    Overwrite(String),
    Append(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleCommand {
    pub args: Vec<String>,
    pub stdin_redirect: Option<String>,
    pub stdout_redirect: Option<RedirectOut>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Connector {
    End,
    Seq,
    And,
    Or,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineItem {
    pub commands: Vec<SimpleCommand>,
    pub connector: Connector,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Word(String),
    Pipe,
    Semi,
    AndIf,
    OrIf,
    RedirectOut,
    RedirectAppend,
    RedirectIn,
}

fn expand_var(
    chars: &[char],
    idx: &mut usize,
    env: &BTreeMap<String, String>,
    last_exit: i32,
) -> String {
    if *idx >= chars.len() {
        return String::from("$");
    }
    if chars[*idx] == '?' {
        *idx += 1;
        return format!("{}", last_exit);
    }
    if chars[*idx] == '{' {
        *idx += 1;
        let start = *idx;
        while *idx < chars.len() && chars[*idx] != '}' {
            *idx += 1;
        }
        let name: String = chars[start..*idx].iter().collect();
        if *idx < chars.len() && chars[*idx] == '}' {
            *idx += 1;
        }
        if name == "?" {
            return format!("{}", last_exit);
        }
        return env.get(&name).cloned().unwrap_or_default();
    }
    let start = *idx;
    while *idx < chars.len() && (chars[*idx].is_ascii_alphanumeric() || chars[*idx] == '_') {
        *idx += 1;
    }
    if *idx == start {
        return String::from("$");
    }
    let name: String = chars[start..*idx].iter().collect();
    env.get(&name).cloned().unwrap_or_default()
}

fn tokenize(
    input: &str,
    env: &BTreeMap<String, String>,
    last_exit: i32,
) -> Result<Vec<Token>, ParseError> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        // Skip whitespace
        if chars[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Comment at start of token
        if chars[i] == '#' {
            break;
        }

        // Operators
        match chars[i] {
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    tokens.push(Token::OrIf);
                    i += 2;
                } else {
                    tokens.push(Token::Pipe);
                    i += 1;
                }
                continue;
            }
            '&' if i + 1 < chars.len() && chars[i + 1] == '&' => {
                tokens.push(Token::AndIf);
                i += 2;
                continue;
            }
            ';' => {
                tokens.push(Token::Semi);
                i += 1;
                continue;
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '>' {
                    tokens.push(Token::RedirectAppend);
                    i += 2;
                } else {
                    tokens.push(Token::RedirectOut);
                    i += 1;
                }
                continue;
            }
            '<' => {
                tokens.push(Token::RedirectIn);
                i += 1;
                continue;
            }
            _ => {}
        }

        // Word token (may combine quoted and unquoted segments)
        let mut word = String::new();
        while i < chars.len() {
            let c = chars[i];
            if c.is_ascii_whitespace() || matches!(c, '|' | ';' | '>' | '<') {
                break;
            }
            if c == '&' && i + 1 < chars.len() && chars[i + 1] == '&' {
                break;
            }
            match c {
                '\'' => {
                    i += 1;
                    let mut closed = false;
                    while i < chars.len() {
                        if chars[i] == '\'' {
                            i += 1;
                            closed = true;
                            break;
                        }
                        word.push(chars[i]);
                        i += 1;
                    }
                    if !closed {
                        return Err(ParseError::UnterminatedQuote);
                    }
                }
                '"' => {
                    i += 1;
                    let mut closed = false;
                    while i < chars.len() {
                        if chars[i] == '"' {
                            i += 1;
                            closed = true;
                            break;
                        }
                        if chars[i] == '\\' && i + 1 < chars.len() {
                            let next = chars[i + 1];
                            match next {
                                '"' | '\\' | '$' => {
                                    word.push(next);
                                    i += 2;
                                    continue;
                                }
                                'n' => {
                                    word.push('\n');
                                    i += 2;
                                    continue;
                                }
                                't' => {
                                    word.push('\t');
                                    i += 2;
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        if chars[i] == '$' {
                            i += 1;
                            word.push_str(&expand_var(&chars, &mut i, env, last_exit));
                            continue;
                        }
                        word.push(chars[i]);
                        i += 1;
                    }
                    if !closed {
                        return Err(ParseError::UnterminatedQuote);
                    }
                }
                '\\' => {
                    i += 1;
                    if i < chars.len() {
                        word.push(chars[i]);
                        i += 1;
                    }
                }
                '$' => {
                    i += 1;
                    word.push_str(&expand_var(&chars, &mut i, env, last_exit));
                }
                _ => {
                    word.push(c);
                    i += 1;
                }
            }
        }
        tokens.push(Token::Word(word));
    }

    Ok(tokens)
}

pub fn parse_line(
    input: &str,
    env: &BTreeMap<String, String>,
    last_exit: i32,
) -> Result<Vec<PipelineItem>, ParseError> {
    let tokens = tokenize(input, env, last_exit)?;
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut items = Vec::new();
    let mut current_pipeline: Vec<SimpleCommand> = Vec::new();
    let mut current_cmd = SimpleCommand {
        args: Vec::new(),
        stdin_redirect: None,
        stdout_redirect: None,
    };

    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            Token::Word(w) => {
                current_cmd.args.push(w.clone());
                i += 1;
            }
            Token::RedirectOut => {
                i += 1;
                match tokens.get(i) {
                    Some(Token::Word(target)) => {
                        current_cmd.stdout_redirect = Some(RedirectOut::Overwrite(target.clone()));
                        i += 1;
                    }
                    _ => return Err(ParseError::MissingRedirectTarget),
                }
            }
            Token::RedirectAppend => {
                i += 1;
                match tokens.get(i) {
                    Some(Token::Word(target)) => {
                        current_cmd.stdout_redirect = Some(RedirectOut::Append(target.clone()));
                        i += 1;
                    }
                    _ => return Err(ParseError::MissingRedirectTarget),
                }
            }
            Token::RedirectIn => {
                i += 1;
                match tokens.get(i) {
                    Some(Token::Word(target)) => {
                        current_cmd.stdin_redirect = Some(target.clone());
                        i += 1;
                    }
                    _ => return Err(ParseError::MissingRedirectTarget),
                }
            }
            Token::Pipe => {
                if current_cmd.args.is_empty() {
                    return Err(ParseError::UnexpectedToken(String::from("|")));
                }
                current_pipeline.push(current_cmd);
                current_cmd = SimpleCommand {
                    args: Vec::new(),
                    stdin_redirect: None,
                    stdout_redirect: None,
                };
                i += 1;
                if i >= tokens.len() {
                    return Err(ParseError::UnexpectedToken(String::from("|")));
                }
            }
            Token::Semi | Token::AndIf | Token::OrIf => {
                let conn = match &tokens[i] {
                    Token::Semi => Connector::Seq,
                    Token::AndIf => Connector::And,
                    Token::OrIf => Connector::Or,
                    _ => unreachable!(),
                };
                if !current_cmd.args.is_empty() {
                    current_pipeline.push(current_cmd);
                    current_cmd = SimpleCommand {
                        args: Vec::new(),
                        stdin_redirect: None,
                        stdout_redirect: None,
                    };
                } else if current_pipeline.is_empty() {
                    let tok_str = match conn {
                        Connector::Seq => ";",
                        Connector::And => "&&",
                        Connector::Or => "||",
                        Connector::End => "",
                    };
                    return Err(ParseError::UnexpectedToken(String::from(tok_str)));
                }
                items.push(PipelineItem {
                    commands: current_pipeline,
                    connector: conn,
                });
                current_pipeline = Vec::new();
                i += 1;
            }
        }
    }

    if !current_cmd.args.is_empty() {
        current_pipeline.push(current_cmd);
    }
    if !current_pipeline.is_empty() {
        items.push(PipelineItem {
            commands: current_pipeline,
            connector: Connector::End,
        });
    }

    Ok(items)
}

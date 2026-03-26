use grafeo::Value;

pub(super) fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.to_string(),
        other => strip_quotes(&other.to_string()),
    }
}

pub(super) fn value_to_f64(v: &Value) -> f64 {
    if let Some(f) = v.as_float64() {
        return f;
    }
    if let Some(i) = v.as_int64() {
        return f64::from(i32::try_from(i).unwrap_or(0));
    }
    0.0
}

fn strip_quotes(s: &str) -> String {
    s.trim_matches('"').trim_matches('\'').to_owned()
}

pub(super) fn format_vector(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|f| format!("{f}")).collect();
    format!("[{}]", parts.join(", "))
}

pub(super) fn gql_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

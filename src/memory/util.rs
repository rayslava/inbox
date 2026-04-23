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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_to_string_string_variant() {
        assert_eq!(value_to_string(&Value::String("hello".into())), "hello");
    }

    #[test]
    fn value_to_string_non_string_strips_quotes() {
        // Int variants stringify without surrounding quotes.
        assert_eq!(value_to_string(&Value::Int64(42)), "42");
    }

    #[test]
    fn value_to_f64_from_float() {
        assert!((value_to_f64(&Value::Float64(1.5)) - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn value_to_f64_from_int() {
        assert!((value_to_f64(&Value::Int64(7)) - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn value_to_f64_fallback_zero() {
        assert!((value_to_f64(&Value::String("not a number".into()))).abs() < f64::EPSILON);
    }

    #[test]
    fn format_vector_formats_comma_separated() {
        let v = vec![1.0_f32, 2.5, -3.25];
        assert_eq!(format_vector(&v), "[1, 2.5, -3.25]");
    }

    #[test]
    fn format_vector_empty() {
        let v: Vec<f32> = vec![];
        assert_eq!(format_vector(&v), "[]");
    }

    #[test]
    fn gql_escape_backslashes_and_quotes() {
        assert_eq!(gql_escape("it's \\ fine"), r"it\'s \\ fine");
    }

    #[test]
    fn gql_escape_plain_string_passes_through() {
        assert_eq!(gql_escape("plain"), "plain");
    }
}

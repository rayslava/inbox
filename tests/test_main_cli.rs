use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn hash_password_command_generates_argon2_hash() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_inbox"))
        .arg("hash-password")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn inbox binary");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(b"super-secret\n").expect("write stdin");
    }

    let output = child.wait_with_output().expect("wait output");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let hash = stdout.trim();
    assert!(
        hash.starts_with("$argon2"),
        "unexpected hash output: {hash}"
    );
}

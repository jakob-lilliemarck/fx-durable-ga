use clap::{CommandFactory, Subcommand};
use clap_complete::aot::{Shell, generate};
use std::env;
use std::fs;
use std::io::Write;

#[derive(Subcommand)]
pub enum CompletionsCommand {
    /// Print completion script to stdout
    Print { shell: Shell },
    /// Install completions (auto-detects shell)
    Install,
}

fn detect_shell() -> Option<Shell> {
    env::var("SHELL").ok().and_then(|shell_path| {
        let shell_name = shell_path.split('/').last()?;
        match shell_name {
            "bash" => Some(Shell::Bash),
            "zsh" => Some(Shell::Zsh),
            "fish" => Some(Shell::Fish),
            _ => None,
        }
    })
}

fn get_completion_dir(shell: Shell) -> anyhow::Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;

    match shell {
        Shell::Bash => Ok(home.join(".bash_completions")),
        Shell::Zsh => {
            // Prefer oh-my-zsh if it exists
            let oh_my_zsh = home.join(".oh-my-zsh/completions");
            if oh_my_zsh.exists() {
                Ok(oh_my_zsh)
            } else {
                Ok(home.join(".zsh/completions"))
            }
        }
        Shell::Fish => {
            let config_dir = dirs::config_dir()
                .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?;
            Ok(config_dir.join("fish/completions"))
        }
        _ => anyhow::bail!("Unsupported shell"),
    }
}

fn install_completion(shell: Shell) -> anyhow::Result<()> {
    let completion_dir = get_completion_dir(shell)?;
    fs::create_dir_all(&completion_dir)?;

    let filename = match shell {
        Shell::Bash => "ga",
        Shell::Zsh => "_ga",
        Shell::Fish => "ga.fish",
        _ => "ga",
    };

    let completion_file = completion_dir.join(filename);
    let mut file = fs::File::create(&completion_file)?;

    generate(shell, &mut super::Cli::command(), "ga", &mut file);

    println!(
        "✓ Installed {} completions to {}",
        shell,
        completion_file.display()
    );

    // Add sourcing to shell rc file for bash
    if matches!(shell, Shell::Bash) {
        let home = dirs::home_dir().unwrap();
        let bashrc = home.join(".bashrc");
        let source_line = "for f in ~/.bash_completions/*; do [ -f \"$f\" ] && source \"$f\"; done";

        if let Ok(contents) = fs::read_to_string(&bashrc) {
            if !contents.contains(source_line) {
                let mut f = fs::OpenOptions::new().append(true).open(&bashrc)?;
                writeln!(f, "\n# Shell completions")?;
                writeln!(f, "{}", source_line)?;
                println!("✓ Added completion sourcing to ~/.bashrc");
            }
        }
    }

    println!("Restart your shell or source your shell config to activate completions");

    Ok(())
}

impl CompletionsCommand {
    pub async fn execute(self) -> anyhow::Result<()> {
        match self {
            CompletionsCommand::Print { shell } => {
                generate(
                    shell,
                    &mut super::Cli::command(),
                    "ga",
                    &mut std::io::stdout(),
                );
                Ok(())
            }
            CompletionsCommand::Install => {
                let shell = detect_shell().unwrap_or_else(|| {
                    println!("Could not detect shell, defaulting to bash");
                    Shell::Bash
                });
                println!("Detected shell: {}", shell);
                install_completion(shell)
            }
        }
    }
}

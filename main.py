#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from summarizer.downloader import download_audio
from summarizer.transcriber import Transcriber
from summarizer.summarizer import Summarizer

console = Console()


def summarize(
    url: str,
    model_size: str = "base",
    max_length: int = 500,
    output: Optional[Path] = None,
    save_transcript: bool = False
):
    """
    Summarize a YouTube video using offline AI models.

    Args:
        url: YouTube video URL
        model_size: Whisper model size (tiny/base/small/medium)
        max_length: Maximum summary length
        output: Save summary to file
        save_transcript: Save full transcript
    """
    console.print(Panel.fit(
        f"[bold blue]🎥 YouTube Video Summarizer[/bold blue]\n"
        f"URL: {url}\n"
        f"Model: Whisper {model_size}",
        border_style="blue"
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # Step 1: Download audio
            task1 = progress.add_task("⬇️  Downloading audio...", total=None)
            audio_path = download_audio(url)
            progress.update(task1, completed=True)
            console.print(f"[green] Audio downloaded: {audio_path.name}[/green]")

            # Step 2: Transcribe
            task2 = progress.add_task("🎤 Transcribing audio...", total=None)
            transcriber = Transcriber(model_size=model_size)
            transcript = transcriber.transcribe(audio_path)
            progress.update(task2, completed=True)

            word_count = len(transcript.split())
            console.print(f"[green] Transcription complete: {word_count} words[/green]")

            # Always save transcript with video ID
            video_id = audio_path.stem
            transcript_file = Path(f"transcript_{video_id}.txt")
            transcript_file.write_text(transcript)
            console.print(f"[blue] Transcript saved: {transcript_file}[/blue]")

            # Step 3: Summarize
            task3 = progress.add_task("Generating summary...", total=None)
            summarizer = Summarizer()
            summary = summarizer.summarize(transcript, max_length=max_length)
            progress.update(task3, completed=True)

        # Display summary
        console.print("\n")
        console.print(Panel(
            summary,
            title="[bold green] Summary[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))

        # Always save summary with video ID
        video_id = audio_path.stem
        summary_file = output or Path(f"summary_{video_id}.txt")
        summary_file.write_text(summary)
        console.print(f"\n[blue] Summary saved to: {summary_file}[/blue]")

        # Cleanup temp audio file
        if audio_path.exists():
            audio_path.unlink()

        console.print("\n[bold green] Done![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red] Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


def info():
    """Display information about available models and system requirements."""
    console.print(Panel.fit(
        "[bold]Available Whisper Models:[/bold]\n"
        "• tiny   - Fastest, least accurate (~39M params, ~75MB)\n"
        "• base   - Good balance (~74M params, ~140MB) [default]\n"
        "• small  - Better accuracy (~244M params, ~460MB)\n"
        "• medium - High accuracy (~769M params, ~1.5GB)\n\n"
        "[bold]Summarization Model:[/bold]\n"
        "• facebook/bart-large-cnn (~400M params, ~1.6GB)\n\n"
        "[bold]System Requirements:[/bold]\n"
        "• Python 3.11+\n"
        "• ~2GB free disk space for models\n"
        "• ~4GB RAM for inference",
        title="Model Information",
        border_style="cyan"
    ))


def main():
    """Main entry point with proper argument parsing."""
    app = typer.Typer(help="Offline YouTube Video Summarizer using AI")

    app.command(name="summarize")(summarize)
    app.command(name="info")(info)

    # Make summarize the default command
    app()


if __name__ == "__main__":
    import sys

    # If first arg looks like a URL, prepend "summarize" command
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        sys.argv.insert(1, "summarize")

    main()

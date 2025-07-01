# TubeScribe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Powered by uv](https://img.shields.io/badge/powered%20by-uv-green.svg)](https://github.com/astral-sh/uv)

An MCP server that provides a tool to fetch transcripts from YouTube videos. It first attempts to
retrieve a pre-existing, official transcript. If one is not available, it downloads the video's audio and uses OpenAI's
Whisper model for local AI-powered transcription.

This project is designed to be a simple, self-contained tool that can be easily integrated into any system capable of
communicating with an MCP server.

## Features

* **YouTube Video Search:** Finds the most relevant YouTube video based on a text query.
* **Official Transcript Priority:** Intelligently fetches manually created or auto-generated YouTube transcripts first
  for speed and accuracy.
* **AI-Powered Fallback:** If no official transcript exists, it automatically falls back to using OpenAI's Whisper
  `tiny` model to generate a high-quality transcript from the video's audio.
* **MCP Server Interface:** Exposes the transcription functionality as a simple tool (`get_youtube_transcript`) via the
  lightweight model context protocol.

## Requirements

* Python 3.12+
* **[uv](https://github.com/astral-sh/uv):** A fast Python package installer and resolver. You will need to [install
  `uv`](https://github.com/astral-sh/uv#installation) on your system first.
* **[FFmpeg](https://ffmpeg.org/download.html):** Must be installed and available in your system's PATH. OpenAI Whisper
  requires it to process audio files.

## Installation with `uv`

Using `uv` is recommended as it's extremely fast and handles both environment creation and package installation
seamlessly.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/YouTubeTranscriber.git
   cd YouTubeTranscriber
   ```

2. **Create and activate a virtual environment:**
   This command creates a `.venv` folder in your project directory and activates it. `uv` will automatically use this
   environment for all subsequent commands.
   ```bash
   uv venv
   ```

3. **Install the project and its dependencies:**
   This command reads the `pyproject.toml` file and installs all required libraries into the virtual environment.
   ```bash
   uv sync
   ```

## Usage

### Running the MCP Server

Once installed, you can start the server by running the `mcp_server.py` script. The server will listen for JSON-RPC
requests on `stdin` and send responses to `stdout`.

```bash 
python mcp_server.py
``` 

The server will log its activity to a file named `mcp_server.log` in the project's root directory.

## Connecting to Gemini CLI on Windows

You can connect this MCP server to the Google Gemini CLI to use the function as a native tool directly from your
terminal. These instructions are for a **Windows** environment.

### Step 1: Create a Startup Script `run_server.bat`

The Gemini CLI needs a single, reliable command to start your server. A batch script is the perfect way to handle this
on Windows, as it ensures the correct virtual environment and Python interpreter are used.

1. Create a new file named in the root of your project directory. `run_server.bat`
2. Copy and paste the following content into the file:

``` batch
    @echo off
    REM This ensures the script's directory is the current directory
    cd /d "%~dp0"
    
    REM --- IMPORTANT ---
    REM Replace the path below with the ABSOLUTE path to your project's venv python.exe
    set PYTHON_EXE="C:\Users\jackp\.pyenv\pyenv-win\versions\3.12.10\python3.12.exe"
    
    echo --- Starting MCP Server using %PYTHON_EXE% ---
    %PYTHON_EXE% mcp_server.py
    
    pause
```

_This script activates the virtual environment in your project and then runs the server, ensuring all the correct
dependencies are available.`.venv`_

### Step 2: Configure the Gemini CLI

Now, you need to tell the Gemini CLI how to find and run your new server.

1. Locate your Gemini CLI `config.json` file. On Windows, this is typically found at:
   `C:\Users\<Your-Username>\.gemini\config.json`
2. Open the `config.json` file in a text editor. Add the following entry to the `mcpServers` object. If `mcpServers`
   doesn't exist, create it as shown below.

``` json
    {
      "mcpServers": {
        "TubeScribe": {
          "command": "C:\\Windows\\System32\\cmd.exe",
          "args": [
            "/c",
            "<path-to-your-project>\\run_server.bat"
          ],
          "cwd": "<path-to-your-project>"
        }
      }
    }
```

3. **Crucially, you must replace both instances of `<path-to-your-project>`** with the full, absolute path to where you
   cloned the `YouTubeTranscriber` repository.

**Example:** If your project is located at `C:\dev\YouTubeTranscriber`, the entry would look like this:

``` json
    {
      "mcpServers": {
        "TubeScribe": {
          "command": "C:\\Windows\\System32\\cmd.exe",
          "args": [
            "/c",
            "C:\\dev\\YouTubeTranscriber\\run_server.bat"
          ],
          "cwd": "C:\\dev\\YouTubeTranscriber"
        }
      }
    }
```

**Note**: JSON requires backslashes to be escaped, so you must use double backslashes (`\\`) in your paths.

### Step 3: Verify the Connection

After saving the `config.json` file, you can verify that Gemini CLI recognizes and can use your new tool.

Run Gemini CLI and press ctrl+t

You should see `TubeScribe` listed as an available tool.

## Connecting to Gemini CLI on Mac/Unix

You can also connect this MCP server to the Google Gemini CLI on Mac or other Unix-like systems. The process is similar to Windows but uses a shell script instead of a batch file.

### Step 1: Prepare the Startup Script

The repository already includes a `run_server.sh` script. Just make it executable:

```bash
chmod +x run_server.sh
```

### Step 2: Configure the Gemini CLI

1. Locate your Gemini CLI `config.json` file. On Mac/Unix systems, this is typically found at:
   `~/.gemini/config.json`

2. Open the `config.json` file in a text editor. Add the following entry to the `mcpServers` object. If `mcpServers`
   doesn't exist, create it as shown below:

```json
{
  "mcpServers": {
    "TubeScribe": {
      "command": "/path/to/your/project/run_server.sh",
      "cwd": "/path/to/your/project"
    }
  }
}
```

3. **Replace both instances of `/path/to/your/project`** with the absolute path to where you cloned the repository.

**Example:** If your project is located at `/Users/username/TubeScribe`, the entry would look like this:

```json
{
  "mcpServers": {
    "TubeScribe": {
      "command": "/Users/username/TubeScribe/run_server.sh",
      "cwd": "/Users/username/TubeScribe"
    }
  }
}
```

### Step 3: Verify the Connection

After saving the `config.json` file, you can verify that Gemini CLI recognizes and can use your new tool.

Run Gemini CLI and press ctrl+t

You should see `TubeScribe` listed as an available tool.

### MCP Client Example

You can interact with the server using any client that supports the MCP protocol over stdio. The server exposes one
primary tool: `get_youtube_transcript`.

Here is an example of a `call_tool` request to get a transcript for the query "What is an API? by MuleSoft".

**Request:**

```json 
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "call_tool",
  "params": {
    "name": "get_youtube_transcript",
    "arguments": {
      "query": "What is an API? by MuleSoft",
      "force_whisper": false
    }
  }
}
``` 

* `query`: The search term for the YouTube video.
* `force_whisper`: (Optional) A boolean that, if `true`, skips the check for an official transcript and generates one
  directly with Whisper. Defaults to `false`.

## Testing

This project includes a test suite to verify its functionality.

* **Core Function Test (`simple.py`):** This script tests the server's handler functions directly without needing to run
  a separate server process. It's the quickest way to check if the core logic is working.
  ```bash
  python simple.py
  ```

* **Full Server Test (`test_mcp.py`):** This script starts the MCP server as a subprocess and sends it live JSON-RPC
  requests, providing an end-to-end test of the server's functionality.
  ```bash
  python test_mcp.py
  ```

## Configuration

* **Logging:** Server activity is logged to `mcp_server.log`.
* **Audio Cache:** When Whisper is used, downloaded audio files are temporarily stored in `testing/audio_cache/`. You
  may wish to change this path in `youtube_tool.py` for a production environment.

## Contributing

Contributions are welcome! If you'd like to improve the YouTube Transcriber, please feel free to fork the repository and
submit a pull request.

Please read our `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

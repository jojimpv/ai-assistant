## AI Assistant
Project integrates local LLM executed via OLLAMA (https://ollama.com/) to a web interface which can act as an AI assistant for form filling tasks. 
This provides user with option to upload the form (PDF file) to be filled in the web interface. 
The form upload is followed by sequence of background tasks like parse the document, auto answer the questions in the form with help of RAG and prompt engineering. 
The solution can be executed in a user’s laptop and make use of open-source technologies.

### OLLAMA Setup

#### OLLAMA Installation
Follow the installation documentation at https://github.com/ollama/ollama

#### OLLAMA model setup
```
ollama pull mxbai-embed-large
ollama pull mistral
ollama pull qwen:4b
```

### Start GUI
1. Execute gui.py

    ```commandline
    cd AI_Assistant/aiassistant
    python3 gui.py
    ```
2. Visit http://localhost:8080/
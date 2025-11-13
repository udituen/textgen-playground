# textgen-playground
Have fun with this text generator by passing in a prompt (the start of a sentence you want to be completed), select the model, decoding strategy and how creative you want the generated text to be

______________________________________________________
### Tech Stack
- FastAPI
- Docker
- Streamlit
- GPT2
- HuggingFace

_____________________________________________________

### Project Folder Structure

project-root/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic models
│
├── streamlit_app.py         # Main Streamlit app (entry point)
│
├── requirements.txt         # Python dependencies
│
├── Dockerfile              # For local development
├── docker-compose.yml      # Optional: orchestrate services locally
│
├── .streamlit/
│   └── config.toml         # Streamlit configuration
│
└── README.md


_________________________________________________________________________

## Resources

- https://github.com/docker/compose?tab=readme-ov-file
- https://docs.streamlit.io/deploy/tutorials/docker


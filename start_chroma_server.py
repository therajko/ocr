import chromadb
from chromadb.config import Settings
from chromadb.server import ChromaServer
import uvicorn

# Configure the server settings
settings = Settings(
    chroma_server_host="0.0.0.0",  # Allow external connections
    chroma_server_http_port=8000,
    allow_reset=True
)

# Create and start the server
server = ChromaServer(settings=settings)

if __name__ == "__main__":
    print("Starting Chroma server on http://0.0.0.0:8000")
    uvicorn.run(
        server.app,
        host=settings.chroma_server_host,
        port=settings.chroma_server_http_port,
        log_level="info"
    ) 
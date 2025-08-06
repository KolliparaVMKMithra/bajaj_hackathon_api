import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# Define the name of the header we will look for
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Load the secret API key from our environment variables
SECRET_KEY = os.environ.get("SECURITY_API_KEY")

async def get_api_key(api_key: str = Security(api_key_header)):
    """Checks if the provided API key is valid."""
    if not SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on the server."
        )
        
    if api_key == SECRET_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key."
        )
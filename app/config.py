from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    blip2_model_name: str = "Salesforce/blip2-opt-2.7b"

settings = Settings()

# In config.py
print(dir())  # See what symbols are available at the end of the file

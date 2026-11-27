"""Configuration non sensible du projet.

Les secrets (cle API) restent dans le .env ; tout le reste vit ici.
"""


BASE_URL = "http://localhost:3001/v1"


TEXT_MODEL = "auto"
VISION_MODEL = "auto"


TIMEOUT_S = 90
MAX_RETRIES = 1

# BLIP Image Captioning API

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.68.0-blue.svg)](https://fastapi.tiangolo.com/)
[![Hugging Face Transformers](https://img.shields.io/badge/transformers-4.19.0-blue.svg)](https://huggingface.co/docs/transformers/index)

docker build -t blip-image-captioning-api .

docker run -d -e JWT_KEY='secret-key' -p 8004:8004 blip-image-captioning-api
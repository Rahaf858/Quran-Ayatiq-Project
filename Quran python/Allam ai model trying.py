from transformers import pipeline

pipe = pipeline("text-generation", model="ALLaM-AI/ALLaM-7B-Instruct-preview", use_auth_token=25)



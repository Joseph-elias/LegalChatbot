import google.generativeai as genai
from dotenv import load_dotenv
import os


genai.configure(api_key="AIzaSyBeQUdqY2x_kuhO-l3zYB--gH5lOOIYwAo")

models = genai.list_models()
for model in models:
    print(model.name, model.supported_generation_methods)

# DaisyUI Component Generator

A FastAPI application that generates HTML components using DaisyUI classes based on user input.

## Features

* Generates HTML components using DaisyUI classes
* Provides a brief explanation for each generated component
* Supports user input for custom component generation
* Utilizes FastAPI for building the backend API
* Integrates with OpenAI for generating components based on user input
* Includes a Jinja2 template for rendering the user interface

## Setup Instructions

1. Set up a virtual environment:
   - Open a terminal in your project folder
   - Run: `python -m venv venv`

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   - Create a file named "requirements.txt" with the following content:
     ```
     fastapi
     uvicorn
     openai
     termcolor
     python-multipart
     jinja2
     python-dotenv
     ```
   - Run: `pip install -r requirements.txt`

4. Run the application:
   - In the terminal, run: `uvicorn main:app --reload`

6. Test the application:
   - Open a web browser and go to: `http://127.0.0.1:8000`

7. Stop the application:
   - Press Ctrl+C in the terminal to stop the uvicorn server

Note: If you're using VS Code, you can also run the application using the debugger:
- Press F5 to start debugging
- Select "FastAPI" from the configuration dropdown
- The application will start, and you can access it in your browser

# Bootstrap Component Generator

A FastAPI application that generates HTML components using Bootstrap classes based on user input.

## Features

* Generates HTML components using Bootstrap 5 classes
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

## Bootstrap Version

This project uses Bootstrap 5.3.3. For documentation and examples, visit the [official Bootstrap website](https://getbootstrap.com/docs/5.3/getting-started/introduction/).

## Customization

To customize Bootstrap's appearance, you can:
1. Use Bootstrap's built-in utility classes
2. Override Bootstrap's default Sass variables
3. Use Bootstrap's mixins to generate your own variants

For more information on customizing Bootstrap, refer to the [official customization guide](https://getbootstrap.com/docs/5.3/customize/overview/).
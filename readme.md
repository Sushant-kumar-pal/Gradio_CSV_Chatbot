# Gradio-based CSV Question Answering and Visualization Application  
![Screenshot 2025-03-13 012600](https://github.com/user-attachments/assets/a0023c13-708a-4860-af89-db43486d6e6e)  

## Overview  
This project is a **Gradio-based application** that enables users to **upload a CSV file, ask questions** (both textual and numerical) about its contents, and receive responses from a **local Large Language Model (LLM)**. Additionally, the application provides **graph plotting capabilities**, ensuring all visualizations are displayed directly within the Gradio interface.  

## Features  
- **CSV File Upload & Handling**  
  - Accepts CSV file uploads with validation.  
  - Handles parsing errors gracefully.  
- **Question Answering via LLM**  
  - Users can input both textual and numerical questions related to the CSV data.  
  - Uses **Ollama + Pydantic AI** to generate answers efficiently.  
  - Recommended LLM model: **Llama 3.2B using Ollama framework**  
- **Graph Plotting & Visualization**  
  - Supports generating **various types of graphs** based on CSV data.  
  - All graphs are embedded **within the Gradio app** interface for seamless interaction.  

## Technical Requirements  
- **Frontend:** [Gradio](https://www.gradio.app/) for an interactive user interface.  
- **LLM Agent:** [Pydantic AI](https://docs.pydantic.dev/latest/) for structured AI query responses.  
- **LLM Backend:** [Ollama](https://ollama.com/) for local execution of language models.  
- **Modular Architecture:**  
  - File handling  
  - Query processing (Pydantic AI)  
  - LLM integration (Ollama)  
  - Graph plotting  
  - Robust error handling for CSV parsing, user input validation, and LLM failures.  

## Installation & Setup  
1. **Clone the Repository:**  
   ```sh
   git clone https://github.com/Adarsh1415/Gradio_based_CSV_Chatbot
   cd Gradio_based_CSV_Chatbot
   ```  
2. **Install Dependencies:**  
   ```sh
   pip install -r requirements.txt
   ```  
3. **Run Llama 3.2 on Ollama Locally:**  
   First, install Ollama following the instructions [here](https://ollama.com/).  
   Then, pull the Llama 3.2 model and run it:  
   ```sh
   ollama pull llama3:2b
   ollama run llama3:2b
   ```  
4. **Run the Application:**  
   ```sh
   python main.py
   ```  

## Notes  
- **Sample CSV:** The application works with datasets like **housing price datasets** from Kaggle, containing both **numerical and string data**.  
- **Max File Size:** 25MB.  

## Working  
![Screenshot 2025-03-13 012623](https://github.com/user-attachments/assets/e0df116f-dfc9-4f90-b0eb-eccf9d2793b3)  

![Screenshot 2025-03-13 012843](https://github.com/user-attachments/assets/f3336660-4bdb-4fbf-a89a-2193923821e1)  

![Screenshot 2025-03-13 013104](https://github.com/user-attachments/assets/d33780b7-20df-42aa-83a3-7dc8f46af355)

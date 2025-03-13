import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from pydantic import BaseModel
from typing import Optional

# -------------------------------
# pydantic_ai Imports and Models
# -------------------------------
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# This model defines the expected structure of the LLM’s output.
class AnalysisResult(BaseModel):
    answer: str

# -------------------------------
# Pydantic Models for the App
# -------------------------------
class UserQuery(BaseModel):
    query: str
    wants_plot: bool = False

# -------------------------------
# CSV Agent using pydantic_ai
# -------------------------------
class Agent:
    """
    A CSV Agent that:
      - Parses user input into a UserQuery.
      - Uses a pydantic_ai LLM agent for inference.
      - Returns a response that might include a textual answer + optional plot.
    """
    def __init__(self, dataframe: pd.DataFrame, model_name: str = "llama3.2"):  # Fixed __init__
        self.df = dataframe
        self.model_name = model_name
        # Initialize the pydantic_ai agent with an OpenAI-compatible model.
        self.llm_agent = PydanticAIAgent(
            OpenAIModel(
                model_name=model_name,
                provider=OpenAIProvider(base_url='http://localhost:11434/v1')
            ),
            result_type=AnalysisResult
        )

    def parse_user_query(self, text: str) -> UserQuery:
        lower_text = text.lower()
        wants_plot = ("plot" in lower_text) or ("graph" in lower_text)
        return UserQuery(query=text, wants_plot=wants_plot)

    def analyze_data(self, user_query: UserQuery) -> str:
        dataset_info = {
            "shape": f"{self.df.shape[0]} rows, {self.df.shape[1]} columns",
            "columns": {},
            "sample_size": min(1000, len(self.df))
        }

        for col in self.df.columns:
            col_info = {
                "dtype": str(self.df[col].dtype),
                "missing_values": self.df[col].isna().sum()
            }
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats = self.df[col].describe()
                col_info.update({
                    "min": stats['min'],
                    "max": stats['max'],
                    "mean": stats['mean'],
                    "std": stats['std'],
                    "percentiles": {
                        "25%": stats['25%'],
                        "50%": stats['50%'],
                        "75%": stats['75%']
                    }
                })
            else:
                col_info.update({
                    "unique_count": self.df[col].nunique(),
                    "most_common": self.df[col].value_counts().nlargest(3).to_dict()
                })
            
            dataset_info["columns"][col] = col_info

        prompt = f"""
        Analyze the FULL DATASET using these precise statistics:
        {dataset_info}
        
        User Query: {user_query.query}
        
        Response Requirements:
        1. Always verify min/max values from provided statistics
        2. Use exact numerical ranges when applicable
        3. Highlight if values approach min/max limits
        4. For range questions, always state absolute boundaries
        
        Format your answer as:
        [Concise Answer]
        [Numerical Evidence: min=..., max=...]
        """
        
        result = self.llm_agent.run_sync(prompt)
        return result.data.answer

    def generate_plot(self):
        numeric_cols = self.df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return None
        col = numeric_cols[0]

        plt.figure()
        self.df[col].hist()

        # Convert the plot to a base64-encoded PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return image_base64

    def run(self, user_message: str):
        user_query = self.parse_user_query(user_message)
        text_answer = self.analyze_data(user_query)
        plot_base64 = None
        if user_query.wants_plot:
            plot_base64 = self.generate_plot()
        return text_answer, plot_base64

# ---------------------------------------
# Global State for CSV and the Agent
# ---------------------------------------
global_df = None
global_agent = None

def upload_csv_action(csv_file):
    """
    Called when a CSV file is uploaded.
    Loads the CSV into a DataFrame and initializes the Agent.
    """
    global global_df, global_agent
    if csv_file is None:
        return "No file uploaded", gr.update(visible=False)
    try:
        df = pd.read_csv(csv_file.name)
        global_df = df
        # Use llama3.2 as the model name
        global_agent = Agent(global_df, model_name="llama3.2")
        return "File uploaded successfully!", gr.update(visible=True)
    except Exception as e:
        return f"Error reading CSV: {str(e)}", gr.update(visible=False)

def chatbot_interface(user_input, chat_history):
    """
    Main chatbot logic: uses the global agent (if available) to process user input.
    """
    if not global_agent:
        bot_msg = "No CSV is loaded yet."
        chat_history.append((user_input, bot_msg))
        return chat_history
    
    text_answer, plot_base64 = global_agent.run(user_input)
    
    if plot_base64:
        bot_msg = text_answer
        chat_history.append((user_input, bot_msg))
        # Add the plot as an image in the Chatbot
        bot_image_msg = f'<img src="data:image/png;base64,{plot_base64}" alt="plot" />'
        chat_history.append(("", bot_image_msg))
    else:
        chat_history.append((user_input, text_answer))
        
    return chat_history

# ---------------------------------------
# Gradio UI Setup
# ---------------------------------------
with gr.Blocks() as demo:
    # Page 1: CSV Upload
    gr.Markdown("## Upload your CSV")
    file_input = gr.File(label="Upload CSV", file_types=[".csv"])
    upload_msg = gr.Markdown(value="", visible=True)
    chat_button = gr.Button("Chat Now", visible=False)
    
    file_input.change(
        fn=upload_csv_action,
        inputs=file_input,
        outputs=[upload_msg, chat_button]
    )
    
    # Page 2: Chat Interface
    with gr.Row(visible=False) as chat_interface:
        chatbot = gr.Chatbot()
        user_textbox = gr.Textbox()
        send_btn = gr.Button("Send")
        
        def show_chat_interface():
            return gr.update(visible=True)
        
        chat_button.click(
            fn=show_chat_interface,
            inputs=None,
            outputs=chat_interface
        )
        
        send_btn.click(
            fn=chatbot_interface,
            inputs=[user_textbox, chatbot],
            outputs=chatbot
        )

demo.launch()

import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")

def predict_churn(usage, tickets, nps, account_age):
    """
    Takes user inputs, creates a single-row DataFrame,
    and returns the predicted churn probability.
    """
    input_data = pd.DataFrame({
        "Usage": [usage],
        "SupportTickets": [tickets],
        "NPS": [nps],
        "AccountAge": [account_age]
    })

    # Get probability of Churned=1
    probabilities = model.predict_proba(input_data)[0]
    prob_churn = probabilities[1]

    return f"Churn Probability: {prob_churn:.2f}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.inputs.Slider(0, 100, step=1, label="Monthly Usage (hours)"),
        gr.inputs.Slider(0, 20, step=1, label="Support Tickets (last quarter)"),
        gr.inputs.Slider(0, 10, step=1, label="NPS (0-10)"),
        gr.inputs.Slider(1, 36, step=1, label="Account Age (months)")
    ],
    outputs="text",
    title="Customer Churn Predictor",
    description="Estimate churn probability from usage, tickets, NPS, and account age."
)

if __name__ == "__main__":
    iface.launch()

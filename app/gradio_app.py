from transformers import pipeline
import gradio as gr

# Load the trained model
classifier = pipeline(
    "text-classification",
    model="models/distilbert-medical-classifier",
    tokenizer="models/distilbert-medical-classifier"
)

def predict(text):
    result = classifier(text)[0]

    return {
        "Predicted Section": result["label"],
        "Confidence (%)": f"{result['score'] * 100:.2f}%"
    }

examples = [
    ["Patients were randomly assigned to two treatment groups."],
    ["The treatment significantly improved survival rates."],
    ["These findings suggest the intervention is effective."],
    ["Cardiovascular disease is a leading cause of death."],
    ["The aim of this study was to evaluate safety."]
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter a medical sentence..."
    ),
    outputs=gr.JSON(),
    title="MedClassifyAI",
    description="Medical text classification using DistilBERT.",
    examples=examples
)

if __name__ == "__main__":
    demo.launch()
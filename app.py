import os
import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def analyze_review(review):
    result = classifier(review)[0]
    return {
        result["label"]: round(result["score"])
    }

demo = gr.Interface(
    fn=analyze_review,
    inputs=gr.Textbox(
        label="Enter your review",
        placeholder="Daliya was good today!"
    ),
    outputs=gr.Label(label="Sentiment"),
    title="VibeCheck AI",
    examples=[
        ["I really enjoyed today!"],
        ["The day was boring."]
    ],
    clear_btn="Clear"
)


demo.launch(
    theme=gr.themes.Soft(),
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)



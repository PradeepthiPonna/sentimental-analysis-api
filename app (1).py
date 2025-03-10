import gradio as gr
from transformers import pipeline

# Load an improved emotion classification model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to predict emotions
def predict_emotions(text):
    if isinstance(text, str):
        text = [text]  # Convert single input to list
    
    results = [classifier(sentence, top_k=3) for sentence in text]  # Get top 3 emotions

    response = []
    for sentence, res in zip(text, results):
        emotions = [f"{r['label']} ({r['score']:.2f})" for r in res]
        response.append(f"📝 Input: {sentence}\n🎭 Predicted Emotions: {', '.join(emotions)}")
    
    return "\n\n".join(response)

# Gradio UI
iface = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(lines=4, placeholder="Enter your text..."),
    outputs=gr.Textbox(label="Emotion Predictions"),
    title="🎭 Emotion Detection App",
    description="Enter a sentence (or multiple) to analyze emotions. Now improved with better predictions!",
    theme="default",
    allow_flagging="never",
    live=True
)

# Launch the Gradio App
iface.launch(share = True)

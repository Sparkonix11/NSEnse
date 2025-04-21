import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer with FinBERT model
        
        Args:
            model_name (str): Name of the HuggingFace model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT has three classes: positive (0), negative (1), neutral (2)
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        
    def analyze(self, texts):
        """
        Analyze sentiment for a list of texts
        
        Args:
            texts (list): List of strings (headlines or news snippets)
            
        Returns:
            list: List of dictionaries with sentiment label and score
        """
        results = []
        
        for text in texts:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get highest probability class
            pred_class = torch.argmax(probs, dim=-1).item()
            
            # Create result dictionary
            result = {
                "text": text,
                "label": self.id2label[pred_class],
                "score": probs[0][pred_class].item(),
                "positive_score": probs[0][0].item(),
                "negative_score": probs[0][1].item(),
                "neutral_score": probs[0][2].item(),
                "sentiment_score": (probs[0][0].item() - probs[0][1].item())  # From -1 to 1 range
            }
            
            results.append(result)
        
        return results
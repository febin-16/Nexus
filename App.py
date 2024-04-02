import torch
from flask import Flask, request, jsonify

# Load your pretrained model
model = torch.load("lstm_model.pt")  # Assuming your model is saved in pt format
model.eval()  # Set model to evaluation mode

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = torch.tensor(data["input"]).float()

        with torch.no_grad():
            output = model(input_data)

        return jsonify({"prediction": output.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Adjust port if needed

import torch
from flask import Flask, request, jsonify
import torch.nn as nn
import numpy as np
app = Flask(__name__)

# Define the LSTM model class
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# Load the LSTM model
input_dim = 5  # Assuming input dimension is 5
hidden_dim = 256  # Assuming hidden dimension is 256
output_dim = 1  # Assuming output dimension is 1
n_layers = 2  # Assuming number of layers is 2
lstm_model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)

# Specify the path to the LSTM model file
model_path = "lstm_model.pt"

# Map tensors to the CPU
device = torch.device("cpu")

# Load the model while mapping tensors to the CPU
lstm_model.load_state_dict(torch.load(model_path, map_location=device))
lstm_model.eval()

# Define route for LSTM predictions
@app.route("/predict/lstm", methods=["POST"])
def predict_lstm():
    try:
        data = request.get_json()
        input_data = np.array(data["input"])  # Assuming input is provided in JSON format
        input_tensor = torch.from_numpy(input_data).to(device).float()

        with torch.no_grad():
            output = lstm_model(input_tensor)

        # Process the output as needed
        prediction = output.tolist()

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

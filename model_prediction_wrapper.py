import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from Model.OCR_Model import OCRModel
import torchvision.transforms.functional as F
import torch.nn.functional as C

def prediction_decode(output):
    probabilities = C.softmax(output, dim=1)
    conf, index_t = torch.max(probabilities, dim=1)
    predicted_index = index_t.item()
    conf_p = conf.item() * 100
    labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
    ]
    predicted_char = labels[predicted_index]

    return predicted_char, conf_p

def predict(image):
        print("[Status] Opening Image...")
        try:
            image_input = Image.open(image).convert("L")

            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1751,), std=(0.3332,))
            ])

            x = transform(image_input).unsqueeze(0)
            x = torch.transpose(2, 3) 

            print(f"[AI] AI is thinking...")
            with torch.no_grad():
                predicted = model(x)

            print(f"[AI] Decoding prediction...")
            predicted_digit, conf = prediction_decode(predicted)
            output = f"[AI] I feel {conf:.2f}% confident that I read the digit {predicted_digit}"
            print(output)
            return output
        except Exception as e:
            print(f"[Error] {e}")
            return "Error"


print("[Status] Loading Model...")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "OCR_Model.pt")
state_dic = torch.load(model_path, weights_only=True)
model = OCRModel()
model.load_state_dict(state_dic)
model.eval()
print("[Info] Model Loaded")

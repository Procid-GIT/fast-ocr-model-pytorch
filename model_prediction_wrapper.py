import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from Model.OCR_Model import OCRModel
import torchvision.transforms.functional as F
import torch.nn.functional as C
from guizero import App, Text, MenuBar
from tkinter import filedialog

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

def predict():
    file_path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        )
    if file_path:
        print("[Status] Opening Image...")
        try:
            image = Image.open(file_path).convert("L")

            
            import numpy as np
            img_np = np.array(image)
            if img_np.mean() > 127: 
                from PIL import ImageOps
                image = ImageOps.invert(image)
                print("[Status] Light image detected. Automatically converted to Dark Mode for the AI!")
            else:
                print("[Status] Native Dark Mode image detected. Processing directly.")
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1751,), std=(0.3332,))
            ])
            
            x = transform(image).unsqueeze(0)

            print(f"[AI] AI is thinking...")
            with torch.no_grad():
                predicted = model(x)

            print(f"[AI] Decoding prediction...")
            predicted_digit, conf = prediction_decode(predicted)
            result.value = f"I feel {conf:.2f}% confident that I saw the digit {predicted_digit}"
            print(f"[AI] I feel {conf:.2f}% confident that I read the digit {predicted_digit}")
        except Exception as e:
            result.value = f"Error :/"
            print(f"[Error] {e}")
    else:
        print("[Status] Cancelled")

print("[Status] Loading Model...")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "OCR_Model.pt")
state_dic = torch.load(model_path, weights_only=True)
model = OCRModel()
model.load_state_dict(state_dic)
model.eval()
print("[Info] Model Loaded")    

app = App("Optical Character Recognizer(Digits)", width=500, height=250)
info = Text(app, text="Welcome to Optical Character Recognizer.  Upload file for recognition.")
info2 = Text(app, text="As of now, this app cannot recognize characters as full sentences, \n like this.  Such changes is for the future to be added.")
menu = MenuBar(app, 
               toplevel=["File"],
               options=[ 
                   [ ["Open", predict] ] 
                ])
result = Text(app, text=" ")

app.display()

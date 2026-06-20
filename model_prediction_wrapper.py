import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from Model.OCR_Model import OCRModel
import torchvision.transforms.functional as F
import torch.nn.functional as C
from guizero import App, Text, MenuBar
from tkinter import filedialog
import numpy as np

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

def slice_sentences():
    file_path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        )
    if file_path:
        try:
            image = Image.open(file_path).convert("L")
            img_np = np.array(image)
            
            if img_np.mean() > 127: 
                from PIL import ImageOps
                image = ImageOps.invert(image)
                print("[Status] Light image detected. Automatically converted to Dark Mode for the AI!")
            else:
                print("[Status] Native Dark Mode image detected. Processing directly.")
            
            img_arr = np.array(image)
            columns = np.sum(img_arr, axis=0)
            ind = np.where(columns > 0)[0]
            if len(ind) == 0:
                return ""
            
            min_col_sum = np.min(columns)
            # Find the max column sum (where text pixels spike)
            max_col_sum = np.max(columns)


            noise_gate = min_col_sum + (max_col_sum - min_col_sum) * 0.1

            seg = []
            in_char = False
            start_column = 0

            for col in range(img_arr.shape[1]):
                has_pixels = columns[col] > noise_gate

                if has_pixels and not in_char:
                    in_char = True
                    start_column = col
                elif not has_pixels and in_char:
                    in_char = False
                    end_col = col
                    seg.append((start_column, end_col))
            
            if not seg:
                return ""
            
            gaps = []
            for i in range(len(seg) - 1):
                gap = seg[i+1][0] - seg[i][1]
                gaps.append(gap)

            average_gap = np.mean(gaps) if gaps else 0
            threshold = average_gap * 1.7

            final = ""

            for i, segment in enumerate(seg):
                left, right = segment
                char_crop = image.crop((left, 0, right, image.height))
                bbox = char_crop.getbbox() # Automatically finds active pixels
                if bbox:
                    char_crop = char_crop.crop(bbox) # Neatly trims top/bottom whitespace
                
                
                char_np = np.array(char_crop)
                char_np = np.where(char_np > 50, 255, 0).astype(np.uint8)
                char_crop = Image.fromarray(char_np)

                w, h = char_crop.size
                max_dim = max(w, h) + 4
                square_img = Image.new('L', (max_dim, max_dim), 0)
                square_img.paste(char_crop, ((max_dim - w) // 2, (max_dim - h) // 2))

                pchar = predict(square_img)

                final += str(pchar)

                if i < len(seg) - 1:
                    live_gap = seg[i+1][0] - right
                    if live_gap > threshold:
                        final += " "

            result.value = f"I predict {final}"

        except Exception as e:
            result.value = f"Error :/"
            print(f"[Error] {e}")
    else:
        print('[Status] Cancelled')
    


def predict(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1751,), std=(0.3332,))
    ])

    x = transform(image).unsqueeze(0)
    x = x.transpose(2, 3)
    print(f"[AI] AI is thinking...")
    with torch.no_grad():
        predicted = model(x)

    print(f"[AI] Decoding prediction...")
    predicted_digit, conf = prediction_decode(predicted)
    res = f"I feel {conf:.2f}% confident that I saw the character {predicted_digit}"
    print(f"[AI] I feel {conf:.2f}% confident that I read the character {predicted_digit}")
    return predicted_digit

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
                   [ ["Open", slice_sentences] ] 
                ])
result = Text(app, text=" ")

app.display()

import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from Model.OCR_Model import OCRModel
import torch.nn.functional as F
import numpy as np

def prediction_decode(output):
    probabilities = F.softmax(output, dim=1)
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
    print(f"RAW MODEL PREDICTED INDEX: {predicted_index}")
    return predicted_char, conf_p

import cv2
import numpy as np

def prepare_for_emnist(char_crop):
    char_np = np.array(char_crop)
    char_np = np.where(char_np > 120, 255, 0).astype(np.uint8)
    char_crop = Image.fromarray(char_np)

    # 2. Get width and height directly using PIL's .size (No .shape error!)
    w, h = char_crop.size
    max_dim = max(w, h) + 4
    
    # 3. Create a perfect black square canvas
    square_img = Image.new('L', (max_dim, max_dim), 0)
    square_img.paste(char_crop, ((max_dim - w) // 2, (max_dim - h) // 2))

    # 4. Resize down to exactly 28x28 for the PyTorch model
    final = square_img.resize((28, 28), Image.Resampling.BILINEAR)
    return final

def slice_sentences(file_path):
    try:
        # 1. Load image and automatically handle light/dark mode
        image = Image.open(file_path).convert("L")
        img_np = np.array(image)
        
        if img_np.mean() > 127: 
            from PIL import ImageOps
            image = ImageOps.invert(image)
            img_np = np.array(image)
            print("[Status] Light image detected. Converted to Dark Mode!")
        
        # 2. Crisp thresholding to isolate characters cleanly
        _, binary_img = cv2.threshold(img_np, 130, 255, cv2.THRESH_BINARY)
        
        # 3. 🔥 THE NEW ENGINE: Find bounding boxes around connected blobs of text
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        seg = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            # Ignore tiny specks of noise that are smaller than 2 pixels wide/tall
            if w > 2 and h > 2:
                seg.append((x, x + w, y, y + h))
        
        if not seg:
            result = "No characters detected."
            return result
            
        # 4. Sort segments from Left to Right (so we read in the correct order!)
        seg.sort(key=lambda x: x[0])

        # 5. Dynamic Word Spacing Logic
        gaps = []
        for i in range(len(seg) - 1):
            gap = seg[i+1][0] - seg[i][1] # distance between current character's right and next character's left
            gaps.append(gap)
        
        average_gap = np.mean(gaps) if gaps else 0
        threshold = average_gap * 1.5 if average_gap > 0 else 10

        # 6. Run inference on each isolated bounding box
        final = ""
        global counter
        counter = 0
        for i, segment in enumerate(seg):
            left, right, top, bottom = segment
            
            # Crop exactly around the character's bounding box coordinates
            char_crop = image.crop((left, top, right, bottom))
            
            # Pad and resize to 28x28 for EMNIST
            square_img = prepare_for_emnist(char_crop)
            pchar = predict(square_img)
            final += str(pchar)

            # Check if we need to add a space between words
            if i < len(seg) - 1:
                live_gap = seg[i+1][0] - right
                if live_gap > threshold and live_gap > 5:
                    final += " "

        predict = f"I predict {final}"
        return predict

        # Clean up global result variable if you use one
        global result_sentence
        result_sentence = final

    except Exception as e:
        error = f"Error :/"
        print(f"[Error] {e}")
        return error

    


def predict(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])
    
    x = transform(image).unsqueeze(0)
    from torchvision.utils import save_image
    global counter
    save_image(x, f'debug{counter}.png')
    counter += 1
    x = torch.rot90(x, k=1, dims=(2, 3))  # Step 1: Rotate 90 degrees counter-clockwise
    x = torch.flip(x, dims=[2])           # Step 2: Mirror it horizontally (dim 3)
    x = x.contiguous()   
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



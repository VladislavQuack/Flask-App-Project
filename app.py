from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torch.nn.modules import BatchNorm2d
import torchvision.transforms as transforms
import pennylane as qml
from PIL import Image
import numpy as np
import cv2
import os
import base64
from io import BytesIO
import torchvision.models as models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load classifictation model
weight_clf_path = r'/workspaces/Flask-App-Project/models/QUANTUM_DENSE_NET_WEIGHTS_BUSIS.pth'
state_dict_clf = torch.load(weight_clf_path, weights_only=True, map_location=device)
DenseNet_FineTuning = models.densenet121(weights=None)
in_features = DenseNet_FineTuning.classifier.in_features
DenseNet_FineTuning.classifier = nn.Linear(in_features=in_features, out_features=3)

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

weight_shapes = {"weights": (4, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = DenseNet_FineTuning
        self.quantum = quantum_layer

    def forward(self, x):
        x = self.cnn(x)
        x = self.quantum(x)
        return x
    
classification_model = QNet()
    
classification_model.load_state_dict(state_dict_clf)
classification_model.eval()

# Load segmentation model
weight_seg_path = r'/workspaces/Flask-App-Project/models/UNEXT_WEIGHTS_BUSIS.pth'
state_dict_seg = torch.load(weight_seg_path, weights_only=True, map_location=device)

class ConvBNReLU(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
  def forward(self, x):
    return self.block(x)

class MLPBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.fc1 = nn.Conv2d(dim, dim, 1)
    self.fc2 = nn.Conv2d(dim, dim, 1)
    self.GELU = nn.GELU()
  def forward(self, x):
    res = x
    x = self.GELU(self.fc1(x))
    x = self.fc2(x)
    return x + res

class UNeXt(nn.Module):
  def __init__(self, in_ch, num_classes=1):
    super().__init__()

    self.enc1 = nn.Sequential(ConvBNReLU(in_ch=in_ch, out_ch=32), ConvBNReLU(32, 32))
    self.pool1 = nn.MaxPool2d(2)

    self.enc2 = nn.Sequential(ConvBNReLU(in_ch=32, out_ch=64), ConvBNReLU(64, 64))
    self.pool2 = nn.MaxPool2d(2)

    self.enc3 = nn.Sequential(ConvBNReLU(in_ch=64, out_ch=128), ConvBNReLU(128, 128))
    self.pool3 = nn.MaxPool2d(2)

    self.bottleneck = nn.Sequential(ConvBNReLU(128, 256), MLPBlock(256))

    self.unpool3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
    self.dec3 = nn.Sequential(ConvBNReLU(256, 128), ConvBNReLU(128, 128))

    self.unpool2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    self.dec2 = nn.Sequential(ConvBNReLU(128, 64), ConvBNReLU(64, 64))

    self.unpool1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
    self.dec1 = nn.Sequential(ConvBNReLU(64, 32), ConvBNReLU(32, 32))

    self.out_conv = nn.Conv2d(32, num_classes, 1)
    # Classification head: 3 classes (benign, malignant, normal)

  def forward(self, x):
    en1 = self.enc1(x)
    en2 = self.enc2(self.pool1(en1))
    en3 = self.enc3(self.pool2(en2))

    bn = self.bottleneck(self.pool3(en3))

    de3 = self.dec3(torch.cat([self.unpool3(bn), en3], dim=1))
    de2 = self.dec2(torch.cat([self.unpool2(de3), en2], dim=1))
    de1 = self.dec1(torch.cat([self.unpool1(de2), en1], dim=1))

    return self.out_conv(de1)
  
segmentation_model = UNeXt(3)

segmentation_model.load_state_dict(state_dict_seg)
segmentation_model.eval()

# Define your class names
CLASS_NAMES = ['ДОБРОКАЧЕСТВЕННОЕ ОБРАЗОВАНИЕ', 'ЗЛОКАЧЕСТВЕННОЕ ОБРАЗОВАНИЕ', 'НОРМАЛЬНОЕ ОБРАЗОВАНИЕ']  # Replace with your actual classes

# Image preprocessing
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay segmentation mask on original image"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert mask to RGB (assuming mask is single channel)
    if len(mask.shape) == 2:
        # Create colored mask (you can customize colors)
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask > 0.5] = [255, 0, 0]  # Red color for segmentation
    else:
        colored_mask = mask
    
    # Resize mask to match image size
    colored_mask = cv2.resize(colored_mask, (img_array.shape[1], img_array.shape[0]))
    
    # Blend image and mask
    overlayed = cv2.addWeighted(img_array, 1-alpha, colored_mask, alpha, 0)
    
    return overlayed

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    image_pil = Image.fromarray(image_array.astype('uint8'))
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data from frontend
        file = request.files['file']
        model_type = request.form.get('type') # 'breast', 'brain', 'spine'
        
        if not file or not model_type:
            return jsonify({'error': 'Missing file or model type'}), 400

        # Read image
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess_image(image)
        
        response_data = {}

        # --- BREAST ANALYSIS (Segmentation + Classification) ---
        if model_type == 'breast' and segmentation_model and classification_model:
            # Segmentation
            with torch.no_grad():
                seg_output = segmentation_model(input_tensor)
                seg_mask = torch.sigmoid(seg_output).squeeze()
            
            # Classification
            with torch.no_grad():
                class_output = classification_model(input_tensor)
                probs = torch.softmax(class_output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()
            
            # Overlay (Red color)
            overlayed = overlay_mask_on_image(image, seg_mask, alpha=0.45)
            def get_class(num):
               if num == 0:
                  return 'Benign'
               elif num == 1:
                  return 'Malignant'
               return 'Normal'
            response_data = {
                'overlayed_image': image_to_base64(overlayed),
                'classification': f"{get_class(pred_class)} ({confidence*100:.1f}%)"
            }

        # --- BRAIN ANALYSIS (Segmentation Only) ---
        elif model_type == 'brain' and segmentation_model:
            with torch.no_grad():
                seg_output = segmentation_model(input_tensor)
                seg_mask = torch.sigmoid(seg_output).squeeze()
            
            # Overlay (Blue color)
            overlayed = overlay_mask_on_image(image, seg_mask, alpha=0.45)
            
            response_data = {
                'overlayed_image': image_to_base64(overlayed),
                'classification': None
            }

        # --- SPINE ANALYSIS (Segmentation Only) ---
        elif model_type == 'spine' and segmentation_model:
            with torch.no_grad():
                seg_output = segmentation_model(input_tensor)
                seg_mask = torch.sigmoid(seg_output).squeeze()
            
            # Overlay (Green color)
            overlayed = overlay_mask_on_image(image, seg_mask, alpha=0.45)
            
            response_data = {
                'overlayed_image': image_to_base64(overlayed),
                'classification': None
            }
        
        else:
            return jsonify({'error': 'Model not loaded or invalid type'}), 500

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
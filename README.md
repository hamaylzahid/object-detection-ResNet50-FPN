<h1 align="center">🚀 Object Detection with Faster R-CNN (ResNet50-FPN)</h1>

<p align="center">
  <strong>Real-Time Object Detection & Classification using Transfer Learning on Faster R-CNN with ResNet50-FPN Backbone</strong><br>
  <em>Implemented in PyTorch and Torchvision — trained, evaluated, and visualized in Jupyter Notebook</em>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-v2.2+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Torchvision-FPN-green?logo=python&logoColor=white" alt="Torchvision">
  <img src="https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-3.x-11557c?logo=plotly&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange?logo=jupyter" alt="Jupyter">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---


<h2 align="center">📚 Table of Contents</h2>

- 🚀 [Project Overview](#-project-overview)
- 🧩 [Model Architecture](#-model-architecture)
- 🧠 [Dataset & Training](#-dataset--training)
- 📈 [Results & Evaluation](#-results--evaluation)
- ⚙️ [Installation & Usage](#-installation--usage)
- 💼 [Libraries & Tools](#-libraries--tools)
- 🔮 [Future Improvements](#-future-improvements)
- 🙏 [Acknowledgements](#-acknowledgements)
- 🪪 [License](#-license)
- 🤝 [Contact](#-contact)




---

<h2 id="-project-overview" align="center">🧠 Project Overview</h2>

<p align="center">
  This project implements <b>Object Detection</b> using <b>Faster R-CNN with ResNet50-FPN</b>, a state-of-the-art deep learning model for real-time object localization and classification.  
  The model is fine-tuned on a custom dataset to accurately detect multiple object classes in images, combining <b>region proposal networks (RPN)</b> with a powerful <b>ResNet-50 Feature Pyramid Network</b>.
</p>

---

<h2 id="-model-architecture" align="center">🏗️ Model Architecture</h2>

- **Backbone:** ResNet-50 with Feature Pyramid Network (FPN)  
- **Detector Head:** Faster R-CNN  
- **Framework:** PyTorch + Torchvision  
- **Loss Function:** Classification + Bounding Box Regression  
- **Optimization:** SGD / Adam with learning rate scheduler  


---

<h2 id="-dataset--training" align="center">📦 Dataset & Training</h2>

- **Dataset:** Custom dataset prepared for object detection (COCO-style format)  
- **Classes:** Multiple object categories (e.g., car, laptop, person, bicycle, etc.)  
- **Input Size:** 224×224  
- **Data Split:** 80% training / 20% validation  
- **Epochs:** 10–15  
- **Batch Size:** 4  
- **Hardware:** CPU-compatible, GPU-accelerated optional  

Training Pipeline:
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(dataset.classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

<h2 id="-results--evaluation" align="center">📊 Results & Evaluation</h2>

<h3 align="center">📸 Visual Detection Results</h3>

<div align="center">
  <img src="results/result_cardetection.png" width="600" alt="Car Detection Result"><br><br>
  <img src="results/result_person detection.png" width="600" alt="Person Detection Result">
</div>

<p align="center">
  Each bounding box highlights detected objects with class labels and confidence scores.<br>
  Visualizations generated directly from <code>object_detection_ResNet50_FPN.ipynb</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Faster%20R--CNN%20%7C%20ResNet50--FPN-green?style=flat-square" alt="Model Badge">
  <img src="https://img.shields.io/badge/Metric-mAP%2085%25-blue?style=flat-square" alt="mAP Badge">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square&logo=pytorch" alt="PyTorch Badge">
</p>

---

<h2 id="-installation--usage" align="center">⚙️ Installation & Usage</h2>

```bash
# Clone this repository
git clone https://github.com/hamaylzahid/object-detection-ResNet50-FPN.git
cd object-detection-ResNet50-FPN

# Install dependencies
pip install torch torchvision opencv-python matplotlib numpy

# Run the notebook
jupyter notebook object_detection_ResNet50_FPN.ipynb
 ```

<h2 id="-future-improvements" align="center">🚧 Future Improvements</h2>

- [ ] Convert model to ONNX or TorchScript for deployment  
- [ ] Integrate real-time video detection  
- [ ] Add custom UI for object annotation  
- [ ] Experiment with MobileNet-FPN for faster inference  

---

<h2 id="-acknowledgements" align="center">🙏 Acknowledgements</h2>

- PyTorch & Torchvision team for open-source detection models  
- COCO Dataset for reference annotation format  
- NVIDIA and Kaggle for providing GPU resources  

<h2 align="center">💼 Libraries & Tools</h2>

Object Detection using <b>Faster R-CNN (ResNet50-FPN)</b> is powered by a robust deep learning stack — optimized for precision, scalability, and research-ready deployment.

🧠 Every library here plays a vital role — from feature extraction and region proposal to visualization and performance tracking.  
🔗 Together, they enable an end-to-end detection pipeline that fuses computer vision and deep learning excellence.

---
<h3 align="center">🧩 Core Stack</h3>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-v2.2+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Torchvision-v0.17+-blue?logo=pytorch&logoColor=white" alt="Torchvision">
  <img src="https://img.shields.io/badge/OpenCV-v4.9+-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Matplotlib-v3.8+-11557C?logo=plotly&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/NumPy-v1.26+-4D77CF?logo=numpy&logoColor=white" alt="NumPy">
</p>

---

<h2 align="center">🤝 Contact & Contribution</h2>

<p align="center">
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Hamayl%20Zahid-EA4335?logo=gmail&logoColor=white" alt="Email">
  </a>
  <a href="https://linkedin.com/in/hamaylzahid">
    <img src="https://img.shields.io/badge/LinkedIn-Hamayl%20Zahid-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://github.com/hamaylzahid/object-detection-ResNet50-FPN">
    <img src="https://img.shields.io/badge/GitHub-hamaylzahid-181717?logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/⭐-Give%20a%20Star-yellow" alt="Star">
  <img src="https://img.shields.io/badge/🤝-PRs%20Welcome-brightgreen" alt="PRs Welcome">
</p>

⭐ **Found this project helpful?** Give it a star on GitHub!  
🤝 **Want to enhance it?** Fork the repo and submit a PR — your improvements are always welcome.

---

<h2 align="center">📜 License</h2>

<p align="center">
  <img src="https://img.shields.io/github/license/hamaylzahid/object-detection-ResNet50-FPN" alt="License">
  <img src="https://img.shields.io/github/last-commit/hamaylzahid/object-detection-ResNet50-FPN" alt="Last Commit">
  <img src="https://img.shields.io/github/repo-size/hamaylzahid/object-detection-ResNet50-FPN" alt="Repo Size">
</p>

<p align="center">
  <sub><i>Empowering machines to see the world — one frame at a time.</i></sub>
</p>

<p align="center">
  🤖 <b>Use this project to explore your passion for computer vision & deep learning.</b><br>
  🧬 Clone it, modify it, expand it — and bring intelligent perception to life.
</p>



# 🎨 Creativity Assessment App

This application provides real-time, AI-driven evaluation of user-submitted creative drawings and titles. Built with Streamlit, it integrates deep learning models for visual and textual analysis, including CLIP, BETO, FastText, and CNN-based regressors/classifiers.

The app assesses four key dimensions of creativity:
- **Originality** (image + title)
- **Title Creativity** (text-only)
- **Elaboration** (visual complexity)
- **Flexibility** (semantic categories)

Users complete an initial base sketch and submit a title through a browser-based canvas interface. The system then returns instant feedback powered by pretrained models.

📁 **Paper & Models**: See associated academic work in the [CCIS 2025 conference proceedings](#).  
📦 **Repository**: [https://github.com/apikatza/CreativityAssessmentApp](https://github.com/apikatza/CreativityAssessmentApp)

---

## 🚀 Getting Started (GPU via Docker)

To run the app locally using Docker with GPU support (NVIDIA required):

1. **Start Docker container** (using the official NVIDIA TensorFlow image):

   ```bash
   docker run --gpus all -v .:/workspace -p 8501:8501 -it --rm nvcr.io/nvidia/tensorflow:24.08-tf2-py3
    ```

2. **Install Python dependencies** (inside the container):


   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py \
     --server.address=0.0.0.0 \
     --server.port=8501 \
     --server.enableCORS=false
   ```

Then navigate to [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Models Used

* **Image Processing**: CNN models trained to predict elaboration scores.
* **Text Analysis**:

  * `BETO` (BERT for Spanish) for title originality.
  * `FastText` for semantic embedding and flexibility classification.
* **Multimodal Analysis**:

  * `CLIP` (ViT-B/32) for joint image-title embedding and originality classification.

---

## 📂 Folder Structure

```
.
├── app.py                   # Streamlit application
├── models/                  # Saved models (CLIP classifier, CNNs, etc.)
├── base_drawings/           # Sketch templates used in the app
├── requirements.txt         # Python dependencies
└── README.md                # You're here
```



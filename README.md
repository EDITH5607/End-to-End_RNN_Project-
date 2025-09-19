# 🎬 End-to-End Sentiment Analysis with RNN

[Live Demo](https://4h8tcsnwhe2a4ivpypafbx.streamlit.app/)  
[GitHub Repository](https://github.com/EDITH5607/End-to-End_RNN_Project-)

---

## 🧮 Project Summary
This project implements an **End-to-End Sentiment Analysis pipeline** using a **Recurrent Neural Network (RNN)** on the **IMDB movie reviews dataset**.  
The model classifies reviews as **Positive** or **Negative** and is deployed via **Streamlit** for live predictions.

✅ Why it’s special:
- Covers the **entire ML workflow** → data preprocessing, model building, training, evaluation, and deployment.  
- Demonstrates practical **deep learning with NLP** using TensorFlow/Keras.  
- Showcases **deployment skills** with Streamlit, making the model interactive and accessible online.  

This project demonstrates strong knowledge of **machine learning engineering**, **natural language processing**, and **model deployment** — skills highly relevant for research and applied AI roles.


**What makes this project stand out:**

- Covers **all stages**: data processing → modeling → deployment  
- Uses deep learning (Embeddings + RNN) to handle sequential text data  
- Deployed via Streamlit, making it interactive and production‐friendly  
- Clean structure with notebooks (for experiments), model artifacts (.h5), and frontend code  

---

## 🗂 Repository Structure

Here’s how your repo is organized:

| File / Folder | Purpose |
|---------------|---------|
| `main.py` | Streamlit app: Takes user input, preprocesses text, predicts sentiment |
| `simpleRNN.ipynb` | Notebook where you train the RNN model (experimentation, hyperparameters) |
| `prediction.ipynb` | Demonstrations of example review predictions using the trained model |
| `embeddings.ipynb` | Study of word embeddings (visualization / exploration) |
| `simple_rnn_imdb.h5` | Saved trained model (architecture + weights) |
| `requirements.txt` | Python package dependencies (versions) |
| `README.md` | Project overview + instructions (you’ll update this) |

---

## 🧠 Technical Deep Dive

### Data & Preprocessing

- **Dataset**: IMDB movie reviews dataset (50,000 reviews, binary sentiment labels)  
- **Vocabulary**: Limits to top ~10,000 most frequent words (common approach to reduce input size and avoid rare words)  
- **Encoding**: Each word → integer via `word_index` dictionary; unknown words mapped with default value  
- **Padding/Truncation**: All sequences are standardized to fixed length (e.g. `maxlen = 500`, or whatever you used) using `pad_sequences`  
- **Lowercasing & splitting**: Normalize text by converting to lowercase, then splitting on whitespace  

### Model Architecture

- **Embedding layer**: Turns word indices into dense vectors (you use an embedding dimension, e.g. 128)  
- **SimpleRNN layer**  
  - Probably with a hidden size (units) you set  
  - May employ dropout / recurrent dropout to reduce overfitting  
- **Dense output layer**: Single neuron with sigmoid activation → outputs probability (0 to 1) for “Positive” sentiment  

### Training & Evaluation

- Loss function: Binary crossentropy  
- Optimizer: Adam (or as specified)  
- Training‐validation split: using IMDB’s test set or a split (noted in your notebook)  
- Monitoring of training & validation loss / accuracy curves (you observe overfitting / how dropouts affect loss)  
- Model persistence: saved to `.h5` file so that inference (in the Streamlit app) loads the trained model  

### Deployment

- The `main.py` file is your Streamlit app: takes raw user input, runs preprocessing, loads saved model, returns prediction  
- Hosted via **Streamlit** (Community or cloud) so it’s accessible online without manual server setup  

---

## 📈 Performance & Observations

- **Training Loss / Validation Loss**: (You had something like ~0.20 training loss vs ~0.40 validation loss)  
  - Indicates some overfitting, consistent with using a simple RNN  
- **Impact of Dropout / Recurrent Dropout**:  
  - Increased training loss (because you regularize / reduce ability to over‐memorize)  
  - Helps generalization, visible in validation metrics  
- **Prediction behavior**: For new reviews, code logic checks if `prediction[0][0] > 0.5` → Positive, else Negative  

---

## 🏃 Installation & Running

### Prerequisites

- Python ≥ 3.7  
- Internet (for installing packages & maybe for model downloading)  

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/EDITH5607/End-to-End_RNN_Project-
cd End-to-End_RNN_Project-

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run main.py

# 5. Open browser at localhost (usually http://localhost:8501)

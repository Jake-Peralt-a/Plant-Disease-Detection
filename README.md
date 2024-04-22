# Plant Disease Prediction with Vision Transformer (ViT)

This project aims to predict plant diseases using deep learning techniques, specifically leveraging the power of Vision Transformers (ViT). By analyzing images of plant leaves, the model can accurately classify whether a plant is healthy or infected with a specific disease.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/your_repository.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pretrained model checkpoint (`plant_disease_model.pth`) from https://drive.google.com/drive/folders/1Wfa81ZZaOM4ON2sa7fb9PDpgsNMQzGyf?usp=sharing and place it in the root directory of the repository.

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

5. Upload an image of a plant. The model will predict the class of the plant (e.g., "Bacterial Spot", "Healthy") along with the confidence score.

## Model

The model used in this project is based on the Vision Transformer (ViT) architecture, pretrained on a large-scale dataset. It has been fine-tuned on a dataset of plant images to perform the specific task of disease classification.

## File Structure

- `app.py`: Main Streamlit application script for user interface and prediction.
- `requirements.txt`: List of Python dependencies required to run the application.
- `plant_disease_model.pth`: Pretrained model checkpoint for plant disease classification.
- `README.md`: Instructions and information about the project (this file).


# Plant Disease Prediction

This project aims to predict plant diseases using machine learning techniques. It utilizes a pretrained Vision Transformer (ViT) model for image classification to predict whether a given plant image is healthy or infected with a specific disease.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/your_repository.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pretrained model checkpoint (`plant_disease_model.pth`) and place it in the root directory of the repository.

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

## Credits

- [Streamlit](https://streamlit.io/): For creating the interactive web application.
- [Hugging Face Transformers](https://huggingface.co/transformers/): For providing the Vision Transformer model and easy-to-use interface for deep learning models.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize this template according to your project's specific details and requirements.

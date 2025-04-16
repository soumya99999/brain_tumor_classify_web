# Brain Tumor Classification

This project is a web application for classifying brain tumors from MRI images using a deep learning model. The app is built with Streamlit and uses a TensorFlow-trained model to classify MRI images as either **Glioma** or **No tumor**.

## Features

- Upload MRI images in JPG, JPEG, or PNG format.
- Classify images using a pre-trained TensorFlow model.
- Display the uploaded image along with the prediction and raw probability score.

## Installation

1. Clone the repository or download the project files.

2. It is recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with the following command:

```bash
streamlit run web.py
```

This will open a local web server where you can upload MRI images and get predictions.

## Model Details

- The model used is a TensorFlow Keras model saved as `1024_relu_xception_model.keras`.
- It classifies images into two classes: "Glioma" and "No tumor".
- Input images are resized to 299x299 pixels and normalized before prediction.

## Dependencies

- streamlit
- tensorflow
- numpy
- pillow
- gdown

## License

This project does not specify a license. Please contact the author for more information.

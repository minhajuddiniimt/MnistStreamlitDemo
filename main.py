import streamlit as st
import model as m
import numpy as np
import logging
from PIL import Image

# Configure logging
logging.basicConfig(
    filename="Web.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    st.title("Mnist Model Demo")
    uploaded_file = st.file_uploader("Upload mage", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        logging.info("Upload image successful.")
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Optionally, make predictions
        preprocessed_image = m.preprocessing(image_array)

        # Predict image text
        predicted_value = m.predict(preprocessed_image)
        st.text(f"Predicted value : {np.argmax(predicted_value)}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

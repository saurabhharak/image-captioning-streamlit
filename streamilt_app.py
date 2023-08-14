import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the saved model and its components
saved_model_directory = "content/saved_model"
saved_feature_extractor_directory = "content/saved_feature_extractor"
saved_tokenizer_directory = "content/saved_tokenizer"

saved_model = VisionEncoderDecoderModel.from_pretrained(saved_model_directory)
saved_feature_extractor = ViTImageProcessor.from_pretrained(saved_feature_extractor_directory)
saved_tokenizer = AutoTokenizer.from_pretrained(saved_tokenizer_directory)

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_model.to(device)

# Define prediction parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Define the prediction function
def predict_step(image):
    i_image = Image.open(image)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = saved_feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = saved_model.generate(pixel_values, **gen_kwargs)

    preds = saved_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def main():
    # Streamlit app
    st.set_page_config(
        page_title="⭐ Image Captioning App",
        page_icon="⭐",
        layout="centered"
    )

    st.title("📷 Image Captioning Using Transformers")
    image = None
    # Upload image or provide URL
    st.write("Upload an image 📤 or provide an image URL 🔗:")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or provide an image URL:")

    if uploaded_image is not None:
        image = uploaded_image
    elif image_url:
        image = image_url
    else:
        st.warning("❗ Please upload an image or provide an image URL.")

    if image:
        try:
            caption = predict_step(image)
            st.image(image, caption=f"📝 Predicted caption: {caption}", use_column_width=True)
        except Exception as e:
            st.error("❌ An error occurred while generating the caption.")
            st.error(str(e))

if __name__ == "__main__":
    main()

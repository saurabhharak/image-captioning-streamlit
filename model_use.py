from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


drive_folder = "/content/drive/My Drive/image_captioning_streamlit"

saved_model_directory = f"{drive_folder}/saved_model"
saved_feature_extractor_directory = f"{drive_folder}/saved_feature_extractor"
saved_tokenizer_directory = f"{drive_folder}/saved_tokenizer"
# Define paths to save the components in your Google Drive


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
def predict_step(image_paths):
    """
    Generate predictions for a list of image paths.

    Args:
        image_paths (List[str]): A list of file paths to the images.

    Returns:
        List[str]: A list of predicted strings.

    Raises:
        None

    Examples:
        >>> image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
        >>> predict_step(image_paths)
        ["prediction1", "prediction2"]
    """
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = saved_feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = saved_model.generate(pixel_values, **gen_kwargs)

    preds = saved_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds



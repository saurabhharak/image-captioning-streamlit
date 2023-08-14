# Project : Image Captioning with Transformers

## Overview
Welcome to my Image Captioning project, where I've harnessed the power of transformers to create an engaging image captioning application. 🌟


## Project Description
### Why Transformers? 🚀
Transformers have taken the world of natural language processing by storm, and now they're making waves in computer vision. With their self-attention mechanism, transformers excel at capturing intricate dependencies and relationships within sequences. This makes them ideal for tasks like image captioning, where understanding context is key. By fine-tuning a VisionEncoderDecoderModel, I've enabled it to grasp both the visual content of an image and its corresponding textual description, resulting in contextually rich captions.

### Model Selection 🧠
I chose the **VisionEncoderDecoderModel** for its prowess in melding vision and language capabilities. It skillfully encodes image content and decodes it into coherent captions. This outshines traditional models like **VGG16**, which fall short due to their lack of sequence-to-sequence generation capabilities. Unlike VGG16, which is designed solely for image classification, the VisionEncoderDecoderModel creates seamless natural language captions.

### Data Considerations 📊
Quality and quantity matter in machine learning, and this project is no exception. While VGG16 may suffice for smaller datasets used in image classification, image captioning demands a broader, more diverse dataset. Transformers thrive on ample data to generalize effectively. The dataset used to train the VisionEncoderDecoderModel should encompass a wide array of images and corresponding captions, ensuring the model captures the intricacies of real-world scenarios.

### Implementation Details 🛠️
The heart of this project beats within the Streamlit framework, providing a user-friendly interface for image captioning. The journey involves loading the pre-trained VisionEncoderDecoderModel, along with its accomplices: the ViTImageProcessor for image preprocessing and the AutoTokenizer for tokenization. Whether you upload an image or provide an image URL, the application seamlessly preprocesses the input, crafts a captivating caption, and proudly showcases the original image alongside its anticipated description.

## Conclusion
By marrying transformers and image captioning, I've unlocked a realm of possibilities that outshine conventional methods like VGG16. Transformers thrive on unraveling intricate relationships and context, making them an intuitive choice for tasks that unite vision and language. This project, nestled in my portfolio, highlights my ability to wield cutting-edge techniques, solving real-world challenges at the intersection of machine learning and computer vision.

## Setup and Usage

### Prerequisites
- Python 3.6 or higher
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saurabhharak/image-captioning-streamlit.git
   cd image-captioning-streamlit

# Downloading Specific Folders from Google Drive

To download specific folders from the Google Drive link, follow these steps:

1. Click the [Google Drive Folder link](https://drive.google.com/drive/folders/1_HOOlB0UEbX7ffyKNpXOfm8FSIBepPYD?usp=drive_link).

2. Locate and right-click on the folders you need:
   - `saved_model`
   - `saved_feature_extractor`
   - `saved_tokenizer`

3. Select "Download" from the context menu.

4. Once downloaded, use the folders and their contents in your project.

**Note:** Ensure you're logged into a Google account and have the necessary permissions.

### The required packages include:

- streamlit
- transformers
- torch
- Pillow

Now you're all set to run the project!

For any questions or issues, please don't hesitate to reach out.


Happy coding!

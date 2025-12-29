# Image Captioning AI

**Task 3:** Combine computer vision and NLP to build an image captioning AI using VGG/ResNet and RNN/Transformer models

## Overview

This project implements an advanced image captioning system that combines:
- **Computer Vision (CNN):** VGG16 for feature extraction
- **Natural Language Processing (RNN/LSTM):** For caption generation

## Features

- VGG16 pre-trained model for image feature extraction
- LSTM encoder-decoder architecture for caption generation
- Support for custom image inputs
- Trained on image-caption datasets
- Real-time caption generation

## Architecture

### Encoder (CNN)
- Uses VGG16 pre-trained on ImageNet
- Extracts high-dimensional features from images
- Output: Feature vectors (4096-dimensional)

### Decoder (RNN)
- LSTM-based sequence-to-sequence architecture
- Generates captions word-by-word
- Uses attention mechanisms for better focus on image regions

## Installation

```bash
# Clone the repository
git clone https://github.com/sivadurga-123/image-captioning.git
cd image-captioning

# Install dependencies
pip install -r requirements.txt
```

## Required Libraries

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
pillow>=8.3.0
matplotlib>=3.4.0
```

## Usage

```python
from image_captioner import ImageCaptioner

# Initialize the captioner
captioner = ImageCaptioner()

# Load a pre-trained model
captioner.load_model('path_to_model')

# Generate caption for an image
image_path = 'path_to_image.jpg'
caption = captioner.generate_caption(image_path)
print(f"Caption: {caption}")
```

## Dataset

The model can be trained on popular image captioning datasets:
- **MS COCO** (Complex Objects in Context)
- **Flickr8k** (8,000 images with 5 captions each)
- **Flickr30k** (30,000 images)

## Model Training

```python
# Training the model
captioner.train(
    images_path='path_to_images',
    captions_path='path_to_captions.txt',
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
```

## Performance Metrics

The model is evaluated using:
- **BLEU Score:** Measures n-gram overlap with reference captions
- **METEOR:** Considers synonyms and word order
- **CIDEr:** Caption-specific similarity metric
- **ROUGE:** Recall-oriented evaluation metric

## Example Output

```
Input Image: [dog.jpg]
Generated Caption: "A brown dog is running in the park"
```

## Project Structure

```
image-captioning/
├── image_captioner.py      # Main implementation
├── README.md               # This file
├── requirements.txt        # Dependencies
├── data/                   # Image and caption datasets
├── models/                 # Pre-trained models
└── utils/                  # Helper functions
```

## Demo

[Demo Video Link](https://www.linkedin.com/feed/)

## Future Enhancements

- [ ] Implement Vision Transformer (ViT) for feature extraction
- [ ] Add attention visualization
- [ ] Support for video caption generation
- [ ] Multi-language caption support
- [ ] Real-time caption generation from camera feed

## License

MIT License - See LICENSE file for details

## Author

Siva Durga (sivadurga-123)

## References

1. "Show and Tell: A Neural Image Caption Generator" - Vinyals et al.
2. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" - Xu et al.
3. TensorFlow Image Captioning Tutorial

## Troubleshooting

### Common Issues

**Issue:** Model loading fails
- **Solution:** Ensure TensorFlow and Keras versions are compatible

**Issue:** Out of memory error
- **Solution:** Reduce batch size or image resolution

**Issue:** Poor caption quality
- **Solution:** Train for more epochs or use a larger dataset

## Contact

For questions or support, please open an issue on GitHub.

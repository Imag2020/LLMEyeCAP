# LLMEyeCap: Giving Eyes to Large Language Models

## Model Description

LLMEyeCap is an innovative Novel Object Captioning model aimed at enhancing Large Language Models (LLMs) with vision capabilities. This project leverages a blend of cutting-edge models and techniques to detect novel objects in images, identify their bounding boxes, and generate insightful captions for them.

One of the core innovations is the replacement of traditional classification layers with text generation mechanisms. This novel approach addresses the issue of catastrophic forgetting, enabling the model to learn new objects without unlearning previous ones. Furthermore, the model connects the latent space of the visual data to the hidden dimensions of an LLM's decoder. This makes it possible to train the model on unsupervised video datasets, opening up a plethora of potential applications.


### Features

- **Novel Object Captioning + Bounding Boxes**
- **ResNet50 as a backbone**
- **Customized DETR model for bounding box detection**
- **BERT Tokenizer and GPT-2 for text generation**
- **Replacing classification layers with Transformer Decoder Object Captioning layers**

## Training Data

The model was trained on the following datasets:
  
- VOC Dataset
- COCO 80 
- COCO 91 

Training was carried out for 30 epochs.

## Usage

Here's how to use this model for object captioning:

\`\`\`python

  model = LLMEyeCapModel(num_queries=NUM_QUERIES,vocab_size=vocab_size,pad_token=PAD_TOKEN)
  model = model.to(device)
  state_dict = torch.load("LLMEyeCap_01.bin")
  model.load_state_dict(state_dict)

  def display_image_ds(image_path, bb, ll):
    #print(len(boxes),len(boxes[0]),len(labels),len(labels[0]))
    image = Image.open(image_path).convert('RGB')
    

    fig, ax = plt.subplots(1, 1, figsize=(12, 20))  # Set the figure size
    
    ax.imshow(image)
    # Draw bounding boxes and labels
    
    for box, label in zip(bb[0], cc[0]):
        
        (x, y, w, h) = box
        if (x==0 and y==0 and w==0 and h==0) or label=='na':
            continue
        x*=image.width
        y*=image.height
        w*=image.width
        h*=image.height
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label_str = tokenizer.decode(label, skip_special_tokens=True)
        #print("*",label_str,"*")
        if label_str != 'na':
            ax.text(x-w/2, y-h/2, label_str, color='r', bbox=dict(facecolor='white', edgecolor='r', pad=2),fontsize=18)
  image_paths=["../data/coco91/train2017/000000291557.jpg", "../data/coco91/train2017/000000436027.jpg"]
  for im in image_paths:
    bb,cc= model.generate_caption( im, tokenizer, max_length=20,pad_sos=PAD_SOS)
    display_image_ds(im, bb.to('cpu'), cc.to('cpu'))

\`\`\`

![image/png](https://cdn-uploads.huggingface.co/production/uploads/645364cbf666f76551f93111/D-0KXDrBzuRCjeF3WcLY3.png)

### Results

. See tuto.ipynb file

## Limitations and Future Work

This 0.1 version is a stand alone model for captiong objects on images. It can be uses as it or trained on new objects without "catastrophic forgetting".
Coming the 0.2 version with latent space to connect to hidden dims of LLMs.
Again this model is still in the development phase and we're actively seeking contributions and ideas to enhance its capabilities. If you're interested in contributing, whether it's through code, ideas, or data, we'd love to hear from you.


## Authors

Imed MAGROUNE.

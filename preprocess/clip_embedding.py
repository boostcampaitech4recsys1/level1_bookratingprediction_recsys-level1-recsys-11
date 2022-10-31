import os
import clip
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd

def embed_image(book_df, model, preprocess, device):
    img_vector_df = book_df[['img_path']].drop_duplicates().reset_index(drop = True).copy()
    data_box = []
    for name in tqdm(img_vector_df['img_path']):
        image_path = os.path.join("/opt/ml/input/code/data", name)
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features.squeeze().detach().cpu().numpy()
        data_box.append(image_features)
    
    img_vector_df['image_embed'] = data_box
    book_df = pd.merge(book_df, img_vector_df, on = 'img_path', how = 'left')
    return book_df


def embed_text(book_df, model, device):
    text_vector_df = book_df[['isbn', 'book_title', 'summary']].drop_duplicates().reset_index(drop = True).copy()
    title_box = []
    summary_box = []
    

            
def main():
    root_dir = "/opt/ml/data/books"
    device = "cuda:0"
    print("[LOAD CLIP]")
    model, preprocess = clip.load("ViT-B/32", device = device)

    for book in sorted(os.listdir(os.path.join(root_dir))):
        print(f"[BOOK NAME: {book.split('.')[0]}]")
        book_df = pd.read_csv(os.path.join(root_dir, book))
        book_df = embed_image(book_df, model, preprocess, device)
        
        


if __name__ == '__main__':
    main()
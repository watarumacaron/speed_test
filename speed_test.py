import torch
import time
from timm import create_model
import pandas as pd

def main(half):
    csv_path = 'top10_results-imagenet.csv'
    model_df = pd.read_csv(csv_path)
    image_sizes = model_df['img_size'].values
    model_names = model_df['model'].values

    trans_model_names = []
    trans_img_sizes = []
    cnn_model_names = []
    cnn_img_sizes = []
    for i in range(model_names.shape[0]):
        if 'swin' in model_names[i] or 'beit' in model_names[i] or 'volo' in model_names[i]:
            trans_model_names.append(model_names[i])
            trans_img_sizes.append(image_sizes[i])
        else:
            cnn_model_names.append(model_names[i])
            cnn_img_sizes.append(image_sizes[i])
    cnn_img_sizes =  sorted(list(set(cnn_img_sizes + [256, 384, 512, 756, 1024])))

    times = []
    for i, model_name in enumerate(trans_model_names):
        image_size = trans_img_sizes[i]
        if half:
            model = create_model(model_name).cuda().half().eval()
            img = torch.zeros((1, 3, image_size, image_size)).cuda().half()
        else:
            model = create_model(model_name).cuda().eval()
            img = torch.zeros((1, 3, image_size, image_size)).cuda()
        with torch.no_grad():    
            out = model(img)
            print(f"{model_name}_{image_size}")
            s = time.time()
            for i in range(100):
                _ = model(img)
            e = time.time()
        times.append((e-s)/100)
        print((e-s)/100)

    df_transformer = pd.DataFrame()
    df_transformer['model_name'] = trans_model_names
    df_transformer['img_size'] = trans_img_sizes
    df_transformer['times'] = times

    df_cnn = pd.DataFrame(index=cnn_img_sizes, columns=cnn_model_names)
    for model_name in cnn_model_names:
        for image_size in cnn_img_sizes:
            if half:
                model = create_model(model_name, features_only=True).cuda().half().eval()
                img = torch.zeros((1, 3, image_size, image_size)).cuda().half()
            else:
                model = create_model(model_name, features_only=True).cuda().eval()
                img = torch.zeros((1, 3, image_size, image_size)).cuda()
            
            with torch.no_grad():
                out = model(img)
                print(f"{model_name}_{image_size}")
                s = time.time()
                for i in range(100):
                    _ = model(img)
                e = time.time()
            df_cnn.loc[image_size, model_name] = (e-s)/100
            print((e-s)/100)

    df_transformer.to_csv(f'transformer_speed_{half}.csv')
    df_cnn.to_csv(f'cnn_speed_{half}.csv')

if __name__ == '__main__':
    main(False)
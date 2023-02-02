import gdown
import os
import zipfile

ID = ["1-TzpAD9JjLl1getsqKrWpSzgeIzqSQ1f", "1FHg-pTTO5Q1ytAUo1KKJHjCiW6oX3lDj"]

for i, id in enumerate(ID):
    url = f'https://drive.google.com/uc?id={id}'

    output = f'./data/Litpart{i}.zip'
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(r'./data')

os.rename("./data/segmentations", "./data/seg")

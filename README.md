
<h1 align="center">Kaggle Project</h1>
<h3 align="center" style="display: flex; justify-content: space-between; width: 100%; text-align: center;">AI vs Human</h3>
<p align="center">
  <img src="https://github.com/Pooh555/AI-vs-human-generated-image/blob/main/res/assets/images/kita_AI.jpg" style="width: 45%; display: inline-block;" />
  <img src="https://github.com/Pooh555/AI-vs-human-generated-image/blob/main/res/assets/images/kita_human.jpg" style="width: 45%; display: inline-block;" />
</p>

### About this project
<p>This project is competing in
  <a href="https://www.infomatrix.ro/" target="_blank"> Infomatrix 2025
  </a>
  .
</p>

### Datasets
<p align="left"> The complete training and testing datasets can be downloaded via the links below.
  <br></br>
  Image Datasets:
  <ol>
    <a href="https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset" target="_blank"> AI vs Human-Generated Images Dataset
    </a>
    <br></br>
    <a href="https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset" target="_blank"> Real vs AI Generated Faces Dataset
    </a>
  </ol>
  Audio Datasets:
  <ol>
    <a href="https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset" target="_blank"> AI vs Human-Generated Voice Dataset
    </a>
  </ol>
</p>

### Project Structure (Depth = 3)
```
├── res
│   ├── assets
│   │   └── images
│   └── test_images
├── src
│   ├── audio
│   │   ├── dataset
│   │   └── graphs
│   └── image
│       ├── cropped_faces
│       ├── cropped_faces_dataset
│       ├── cropped_test_faces
│       ├── dataset
│       ├── faces_dataset
│       └── graphs
└── trained_models
```
### Essential packages
Install essential packages
```
conda install python=3.10.9
pip install -r requirements.txt
```

Import necessary libraries and modules including:
| Libraries | Version | Channel |
| ----------- | ----------- | ----------- |
| Keras | 2.11.0 | conda-forge |
| Librosa | 0.11.0 | PyPi |
| Matplotlib | 3.9.1 | conda-forge |
| NumPy | 2.2.2 | conda-forge |
| Pandas | 2.2.3 | conda-forge |
| Pillow | 9.4.0 | conda-forge |
| Pytorch | 2.5.1 | conda-forge |
| Seaborn | 0.13.2 | conda-forge |
| Sklearn | 1.6.1 | conda-forge |
| Tensorflow | 2.18.0 | conda-forge |

### Resources
#### Pooh555's laptop
| Devices | Specification | Remark |
| ----------- | ----------- | ----------- |
| CPU | Ryzen 7 6800HS | - |
| GPU | GeForce RTX 3050 | CUDA 12.7 |
| RAM | - | 16 GB |
| OS | Arch Linux | x84_64 |

#### Pooh555's host PC
| Devices | Specification | Remark |
| ----------- | ----------- | ----------- |
| CPU | i7 gen 11-11700K | 5.0 GHz |
| GPU | GeForce RTX 4060 Ti | CUDA 12.6 |
| RAM | - | 32 GB |
| OS | Ubuntu | - |
#### Using host's resources
Installing required packages\
Ubuntu (bash)
```
curl -s https://install.zerotier.com | sudo bash
sudo apt install openssh-client
```
Arch Linux (bash)
```
curl -s https://install.zerotier.com | sudo bash
sudo pacman -S openssh
```
Connect to Pooh555's Private network, and ask him for authorization.
```
zerotier join 17d709436c99ecf8 
```
Connect to host's terminal via ssh
```
systemctl start sshd
ssh pooh@10.240.139.131
conda activate image_classification
```
After completing these procedures you may open vscode ssh session or run jupyter file via command line.
### Team Members
<ol>
  <li>Pooh555
    <a href="https://github.com/Pooh555" target="_blank"> Github Profile.</a>
  </li>
</ol>
<p align="center">
  We are from Kamoetvidya Science Academy (
  <i>
    <a href="https://www.kvis.ac.th" target="_blank"> KVIS</a>
  </i>
  ).
</p>

# Super Correlation Verification for Image Retrieval
Welcome to an unofficial repository that houses a PyTorch implementation of the image retrieval network presented in the research paper, [Correlation Verification for Image Retrieval](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Correlation_Verification_for_Image_Retrieval_CVPR_2022_paper.html) [1], as well as the modifications proposed in [Global Features are All You Need for Image Retrieval and Reranking](https://arxiv.org/abs/2308.06954) [2]. An official repository for both papers do exist [here](https://github.com/sungonce/CVNet/tree/main) and [here](https://github.com/ShihaoShao-GH/SuperGlobal) respectively, but unfortunately, the repos lacks the comprehensive code necessary for reproducing training results due to [cited](https://github.com/sungonce/CVNet/issues/1#issuecomment-1161781271) intellectual property concerns.

Thus this repository has been built with the objective to bridge this gap by providing a more complete and coherent codebase. Initiative has been taken to include a well-structured easy to follow codebase as well as a clear training loop, aimed to encapsulate the full essence of the networks proposed in the original papers. This in hopes to promote more rapid and straightforward reproducibility and facilitates smoother training transitions on novel datasets.

Note that this repo is still a work in progress. See the to do list. 

## Getting started

### Requirements
After cloning the repository, 
```bash
git clone https://github.com/edwardguil/SuperCVNet.git
```
it is suggested to create a new conda env
```bash
conda create --name supercvnet python=3.12
conda activate supercvnet
```
then install the dependancies from the requirements.txt
```bash
pip install -r requirements.txt
```
### Minimal Usage
The training scripts are contained in train_backbone.py and train_rerank.py. You can run these scripts from the command line, which by default starts a training loop on Cifar10:
```bash
python train_backbone.py 
```
Or by importing the training loop for more control over the inputs to the training proccess:
```python
from train_backbone import train_backbone
train_backbone(...)
```

## Expanded Usage
### CVNet Usage
CVNet is implemented into two distinct classes:
```python
class CVNetGlobal()
    pass

class CVNetRerank()
    pass
```
These models can be used like normal Pytorch models e.g.
```python
from models import CVNetGlobal, CVNetRerank
model = CVNetGlobal()
rerank = CVNetRerank()

x = torch.rand((1, 3, 512, 512))
y = model(x)
y_ranked = rerank(y)
```
For training, as per the paper, CVNet requires positive sample pairs to be passed through the momentum network. To simplify this proccess, you can utilize the PairedDataset class as a wrapper around existing Pytorch datasets. Note that datasets that can be anything, as long as they can be indexed (i.e. have the __get_item__ function implemented)  e.g.
```python
from torchvision.datasets import CIFAR10
from datasets import PairedDataset()
dataset = CIFAR10()
dataset[0] # This dataset is indexable 
paired_dataset = PairedDataset(dataset)
for x, x_positive, y in paired_dataset:
    # Here x and x_positive share the same label (y)
    pass
```

## SuperGlobal Usage
SuperGlobal is also implemented into two distinct classes:
```python
class SuperGlobal()
    pass

class SuperGlobalRerank()
    pass
```
These models can be used together or independantly like normal Pytorch models.
```python
from models import SuperGlobal, SuperGlobalRerank
model = SuperGlobal()
rerank = SuperGlobalRerank(...)

x = torch.rand((1, 3, 512, 512))
y = model(x)
y_ranked = rerank(y)
```
The caveat to the above, is that SuperGlobaRerank requires access to a vector database(db) for similarity search. If you simply want to perform similarity on a tensor of vectors, use the TensorVectorDB class:
```python
from helpers import TensorVectorDB
from models import SuperGlobalRerank

vectors = torch.rand((10*3, 512)) # num vectors x feature dim
labels = torch.rand((10*3, 1)) # num vectors x label dim
vector_db = TensorVectorDB(vector_set, labels)

rerank = SuperGlobalRerank(vector_db)
```
If you want to some other form a vector database, simply implement a child of AbstractVectorDB contained in helpers/base/vector_db.py. There already exists a pinecone_index if you want to use a Pinecone database as your vector store.

## To Do List:
- [x] Implement generic vectordb class to allow for easier extensability
- [x] Implement SuperRerank network 
- [ ] Complete the train_rerank script.
- [ ] Add correct transforms and class count for Google Landmarks
- [ ] Add input args for channel norms and resizing customization


## References 
[1] Lee, S., Seong, H., Lee, S., & Kim, E. (2022). Correlation Verification for Image Retrieval. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5364-5374.
[2] Shao, S., Chen, K., Karpur, A., Cui, Q., Araújo, A.F., & Cao, B. (2023). Global Features are All You Need for Image Retrieval and Reranking. ArXiv, abs/2308.06954.

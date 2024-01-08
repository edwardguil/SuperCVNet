# Super Correlation Verification for Image Retrieval
Welcome to an unofficial repository that houses a PyTorch implementation of the image retrieval network presented in the research paper, [Correlation Verification for Image Retrieval](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Correlation_Verification_for_Image_Retrieval_CVPR_2022_paper.html) [1], as well as the modifications proposed in [Global Features are All You Need for Image Retrieval and Reranking](https://arxiv.org/abs/2308.06954) [2]. An official repository for both papers do exist [here](https://github.com/sungonce/CVNet/tree/main) and [here](https://github.com/ShihaoShao-GH/SuperGlobal) respectively, but unfortunately, the repos lacks the comprehensive code necessary for reproducing training results due to [cited](https://github.com/sungonce/CVNet/issues/1#issuecomment-1161781271) intellectual property concerns.

Thus this repository has been built with the objective to bridge this gap by providing a more complete and coherent codebase. Initiative has been taken to include a well-structured easy to follow codebase as well as a clear training loop, aimed to encapsulate the full essence of the networks proposed in the original papers. This in hopes to promote more rapid and straightforward reproducibility and facilitates smoother training transitions on novel datasets.

Note that this repo is still a work in progress. See the to do list. 

## To Do List:
- Add correct transforms and class count for Google Landmarks
- Add input args for channel norms and resizing customization
- Correct train_rerank.py to get descriptors from Pinecone/CSV file
- Implement SuperRerank network

## References 
[1] Lee, S., Seong, H., Lee, S., & Kim, E. (2022). Correlation Verification for Image Retrieval. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5364-5374.
[2] Shao, S., Chen, K., Karpur, A., Cui, Q., Araújo, A.F., & Cao, B. (2023). Global Features are All You Need for Image Retrieval and Reranking. ArXiv, abs/2308.06954.

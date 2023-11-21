# 2024-TIP-DMLPA
Peng Hu, Liangli Zhen, Xi Peng, Hongyuan Zhu, Jie Lin, Xu Wang, Dezhong Peng, Deep Supervised Multi-View Learning with Graph Priors (IEEE TIP 2024, PyTorch Code).

## Abstract
This paper presents a novel method for supervised multi-view representation learning, which projects multiple views into a latent common space while preserving the discrimination and intrinsic structure of each view. Specifically, an \textit{apriori} discriminant similarity graph is first constructed based on labels and pairwise relationships of multi-view inputs. Then, view-specific networks progressively map inputs to common representations whose affinity approximates the constructed graph. To achieve graph consistency, discrimination, and cross-view invariance, the similarity graph is enforced to meet the following constraints: 1) pairwise relationship should be consistent between the input space and common space for each view; 2) within-class similarity is larger than any between-class similarity for each view; 3) the inter-view samples from the same (or different) classes are mutually similar (or dissimilar). Consequently, the intrinsic structure and discrimination are preserved in the latent common space using an \textit{apriori} approximation schema. Moreover, we present a sampling strategy to approach a sub-graph sampled from the whole similarity structure instead of approximating the graph of the whole dataset explicitly, thus benefiting lower space complexity and the capability of handling large-scale multi-view datasets. Extensive experiments show the promising performance of our method on five datasets by comparing it with 18 state-of-the-art methods.

## Framework
<img src="./fig/framework.png" width = "100%" height="50%">

[![\\ Fig. 1: The framework of DMLPA. In the figure, distinct shapes are used to represent diverse classes and distinct colors are used to denote different views. $\mathbf{W}$ and $\mathbf{V}^{kk}$ are the similarity matrices of all common representations and the $k$-th view inputs $\mathcal{X}_{k}$, respectively. $\mathbf{L}$ and $\mathbf{H}$ are the normalized graph Laplacian matrices that represent the graphs of common space and input data, respectively. Moreover, $\mathbf{L}$ and $\mathbf{H}$ are respectively computed by $\mathbf{W}$ and $\mathbf{V}^{kl}|_{k,l}^{v}$  (see \Cref{L} and \Cref{H}), where $\mathbf{V}^{kl}|_{k \neq l}^{v}$ are inter-view similarity matrices computed by intra-view similarity matrices $\mathbf{V}^{kk}|_{k}^{v}$ and labels (see \Cref{V-item-inter}). $\mathcal{J} = \frac{1}{N} \| \mathbf{H} - \mathbf{L} \|_{F}^{2}$ is the loss to make the obtained common representations approximate \textit{apriori} similarity graph of input data.](https://latex.codecogs.com/svg.latex?%5C%5C%20The%20framework%20of%20DMLPA.%20In%20the%20figure%2C%20distinct%20shapes%20are%20used%20to%20represent%20diverse%20classes%20and%20distinct%20colors%20are%20used%20to%20denote%20different%20views.%20%24%5Cmathbf%7BW%7D%24%20and%20%24%5Cmathbf%7BV%7D%5E%7Bkk%7D%24%20are%20the%20similarity%20matrices%20of%20all%20common%20representations%20and%20the%20%24k%24-th%20view%20inputs%20%24%5Cmathcal%7BX%7D_%7Bk%7D%24%2C%20respectively.%20%24%5Cmathbf%7BL%7D%24%20and%20%24%5Cmathbf%7BH%7D%24%20are%20the%20normalized%20graph%20Laplacian%20matrices%20that%20represent%20the%20graphs%20of%20common%20space%20and%20input%20data%2C%20respectively.%20Moreover%2C%20%24%5Cmathbf%7BL%7D%24%20and%20%24%5Cmathbf%7BH%7D%24%20are%20respectively%20computed%20by%20%24%5Cmathbf%7BW%7D%24%20and%20%24%5Cmathbf%7BV%7D%5E%7Bkl%7D%7C_%7Bk%2Cl%7D%5E%7Bv%7D%24%20%20(see%20%5CCref%7BL%7D%20and%20%5CCref%7BH%7D)%2C%20where%20%24%5Cmathbf%7BV%7D%5E%7Bkl%7D%7C_%7Bk%20%5Cneq%20l%7D%5E%7Bv%7D%24%20are%20inter-view%20similarity%20matrices%20computed%20by%20intra-view%20similarity%20matrices%20%24%5Cmathbf%7BV%7D%5E%7Bkk%7D%7C_%7Bk%7D%5E%7Bv%7D%24%20and%20labels%20(see%20%5CCref%7BV-item-inter%7D).%20%24%5Cmathcal%7BJ%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5C%7C%20%5Cmathbf%7BH%7D%20-%20%5Cmathbf%7BL%7D%20%5C%7C_%7BF%7D%5E%7B2%7D%24%20is%20the%20loss%20to%20make%20the%20obtained%20common%20representations%20approximate%20%5Ctextit%7Bapriori%7D%20similarity%20graph%20of%20input%20data.)](#_)

## Usage
To train a model, just run train.sh:
```bash
sh train.sh
```

## Comparison with the State-of-the-Art
<div align=center>TABLE IV: Comparative results (MAP@ALL) for cross-view retrieval on the Pascal Sentence dataset.</div>
<div align=center><img src="./fig/Table4.png" width = "50%" height="50%"></div>

<div align=center>TABLE V: Comparative results (MAP@ALL) for cross-view retrieval on the XMediaNet dataset.</div>
<div align=center><img src="./fig/table5.png" width = "50%" height="50%"></div>

<div align=center>TABLE VI: Comparative results (MAP@ALL) for cross-view retrieval on the MS-COCO dataset.</div>
<div align=center><img src="./fig/table6.png" width = "50%" height="50%"></div>


## Citation
If you find DMLPA useful in your research, please consider citing:
```
@inproceedings{hu2024deep,
   title={Deep Supervised Multi-View Learning with Graph Priors},
   author={Peng Hu, Liangli Zhen, Xi Peng, Hongyuan Zhu, Jie Lin, Xu Wang, Dezhong Peng},
   booktitle={IEEE Transactions on Image Processing},
   pages={},
   year={2024}
}
```

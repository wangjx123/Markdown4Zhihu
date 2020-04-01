## Learning-based Visual Odometry(1)---Understanding the Limitations of CNN-based Absolute Camera Pose Regression

## 导读：

​		在基于图像的相机重定位（Visual Relocalization）和视觉里程计（Visual Odometry，VO）等任务中，核心问题是求解运动相机自身的姿态，这类问题经典的解决方法是视觉几何模型，这依赖于图像间匹配的特征点。近两年来，大量的基于deep learning的方法兴起，他们大多是构建一个CNN网络用以回归相机的位姿（通常该网络被称为posecnn）。对于相机重定位问题，输入图像大多是单帧图像，输出值为该相机的绝对姿态（Absolute Camera Pose）；对于视觉里程计问题，输入大多为两帧图像，输出值为两幅图像的相对变换关系（Relative Camera Pose）。这种方法的优势是端到端训练，且对于视觉几何基础一般的人（我）而言更好上手。在相机重定位问题中，大多是有监督的训练，即每幅图像都对应一个真实的绝对的位姿标签。在视觉里程计问题中，除了有监督的训练方式外，还有大量的无监督训练方式，但这需要借助深度网络联合训练$^{[4]}$。这些方法在测试集上具有较好的表现，但不及传统的基于特征点匹配的视觉几何方法。而且这些基于CNN回归的方法泛化性较差，以我个人做过的基于无监督的VO为例，如果输入图像顺序一直是前帧到后帧的顺序，当测试图像反序时，PoseCnn仍然输出正向的translation。因此这篇文章基于相机重定位任务，分析了基于CNN的绝对位姿回归方法的局限性，当然对于VO方面的结论也是类似的。 这篇文章理论分析并不难理解，且实验非常充分，而且还配有实验展示视频$^{[2]}$，文中的图像展示也很高大上，总之值得学习啊。



## 正文：

### 1. 贡献：

​		（1）首次理论分析基于CNN回归的绝对位姿估计（Absolute Pose Regression, APR）的局限性，并得出结论：该方法更像是一种图像检索而不是通过3D几何的方式求解。

​		（2）通过理论和实验证明了APR方法泛化性能差。

​		（3）实验表明，在求解相机位姿时，APR与基于人工设计的图像检索方法性能相近，但比基于传统的3D几何方法（Stucture-based）要差。

### 2. 理论证明

​		作者提出CNN将图像映射到位姿的过程分为三个阶段：第一阶段图像编码，通常的backbone为ResNet，VGG等，输入图像为$I$ ,编码后为 $F(\mathcal{I})$ .  第二阶段再将其变换成一高维向量（前两阶段可合并），记作$E(F(\mathcal{I}))$ ，结果为$\alpha^{\mathcal{I}}=\left(\alpha_{1}^{\mathcal{I}}, \ldots, \alpha_{n}^{\mathcal{I}}\right)^{T} \in \mathbb{R}^{n}$ ，最后通过全连接层输出位姿，其数学表达形式如下：



​					$\begin{aligned} L(\mathcal{I}) &=\mathbf{b}+\mathrm{P} \cdot E(F(\mathcal{I})) \\ &=\mathbf{b}+\mathrm{P} \cdot\left(\alpha_{1}^{\mathcal{I}}, \ldots, \alpha_{n}^{\mathcal{I}}\right)^{T} \end{aligned}$     （1）

​	

其中，L表示visual localization function， $\mathrm{P} \in \mathbb{R}^{(3+r) \times n}$ is a projection matrix（全连接层的权重） ，$\mathbf{b} \in \mathbb{R}^{3+r}$是偏置。$L(\mathcal{I})$ 输出 $\hat{\mathbf{p}}_{\mathcal{I}}=$ $\left(\hat{\mathbf{c}}_{\mathcal{I}}, \hat{\mathbf{r}}_{\mathcal{I}}\right)$ 作为相机位姿。

​		如果把全连接层权重$P$的每列分开来看，即 $\mathbf{P}_{j} \in \mathbb{R}^{3+r}$ 表示$j^{\text {th }}$ column of $P $。上面的式子可以改写成$P$各列的线性组合，其权重就是$\alpha^{\mathcal{I}}=\left(\alpha_{1}^{\mathcal{I}}, \ldots, \alpha_{n}^{\mathcal{I}}\right)^{T} \in \mathbb{R}^{n}$，可写成如下表达式：



​					$L(\mathcal{I})=\mathbf{b}+\sum_{j=1}^{n} \alpha_{j}^{\mathcal{I}} \mathbf{P}_{j}=\left(\begin{array}{l}\hat{\mathbf{c}}_{\mathcal{I}} \\ \hat{\mathbf{r}}_{\mathcal{I}}\end{array}\right)$   （2）



​		当然，如果把P中按行来分析，又可以把translation 与 rotation 解耦：



​					$\left(\begin{array}{c}\hat{\mathbf{c}}_{\mathcal{I}} \\ \hat{\mathbf{r}}_{\mathcal{I}}\end{array}\right)=\left(\begin{array}{c}\mathbf{c}_{b}+\sum_{j=1}^{n} \alpha_{j}^{\mathcal{T}} \mathbf{c}_{j} \\ \mathbf{r}_{b}+\sum_{j=1}^{n} \alpha_{j}^{T} \mathbf{r}_{j}\end{array}\right)$        （3）



式（3）中$\mathbf{c}_{j} \in \mathbb{R}^{3}$ ，and  $\mathbf{r}_{j} \in \mathbb{R}^{r},$ 分别表示偏移和旋转部分，如果仅以偏移量部分考虑，作者将之称为base translation. 最终的输出 $\mathbf{P}_{j}=\left(\mathbf{c}_{j}^{T}, \mathbf{r}_{j}^{T}\right)^{T} .$ 类似的， $\mathbf{b}$ as $\mathbf{b}=\left(\mathbf{c}_{b}^{T}, \mathbf{r}_{b}^{T}\right)^{T}$。**这个式子是全文分析的核心**，尤其是base translation这个思想（后文分析不考虑rotation, 可能无论是四元数还是旋转向量都没有直观的表达意义，而且仅分析trans也能表达作者的意图），base translation是网络学出来的，它受到训练图像的影响，表征着训练图像translation的基（一般得，算上rotation也称**位姿基**），$\alpha^{\mathcal{I}}$是每个图像的hidden feature，在作者看来它是base translation线性组合的系数。



### 实验证明

1. **Settings**

   训练数据集与测试数据集中的真值都是由SfM(Sturcture From Motion)方法得到。基于CNN的方法选择了具有代表性的PoseNet和MapNet，基于结构的方法选择了Active Search。Active Search 使用了Root-SIFT特征确定了2D-3D匹配关系，位姿估计由RANSAC循环中的P3P方法得到，再经过一个非线性的优化。$^{[1]}$

2. **训练在一条直线或者多条直线上**

   如果不考虑相机的抖动，此时相机的translation部分应该在一条直线上。那么此时的训练图像作为input时，CNN的base translation是不是就在一条直线上呢？如果测试图像不在这条直线上，是不是就无法预测出准确的pose呢？  如图1所示，这里表示两个场景，两次训练过程。以第一个场景为例（图1左边的四幅图），首先MapNet确实如此，左上角是对base translation可视化，PoseCNN（左下角）似乎噪声更多些。图中深红色的线是训练路径，绿色的线是测试路径。深蓝色的线是MapNet的测试效果，紫色是PoseCnn测试效果，浅蓝色的线是几何法的测试效果。从中基本可以证实作者猜想，当给出与训练数据不同直线下的场景，APR方法失效了，而传统法效果不错。

   <img src="blog1_Learning-based Visual Odometry(1)/截屏2020-04-0111.49.47.png" alt="截屏2020-04-0111.49.47" style="zoom:50%;" />

   ​															   	 图1

3.**一般化的训练场景**

​		照作者的理论，如果训练数据够丰富，场景更多，那么base translation(pose)应该泛化的更好，当然数据确实要够多，这个首先很难保证，其次这是个必要条件。如图2、图3所示，唯有浅蓝色的几何法能在测试集上表现得较好，APR方法还是只学到了训练数据上的base pose，当出现新的场景时（绿色的线），他们依然是对这些base pose的线性组合，特别是当训练图像与测试图像相近的时候pose回归方法能得到较好的值，即红绿线相交处，蓝紫色线很密。作者想，pose回归的方法就是根据测试图像与训练图像的相似度来组合这个pose，从这个角度看，作者想到了图像检索方法。

所有实验内容可参看作者给出的视频 $^{[2]}$ ，很有意思。

![截屏2020-04-0111.53.03](blog1_Learning-based Visual Odometry(1)/截屏2020-04-0111.53.03.png)

​																		图2

![截屏2020-04-0111.54.01](blog1_Learning-based Visual Odometry(1)/截屏2020-04-0111.54.01.png)

​																	图3	

作者对图2和图3做了一个总结$^{[1]}$:

​		（1）所有的平移基都在一个平面上，因为训练图像的拍摄位置都在同一个平面上；
​		（2）当测试轨迹与训练轨迹相交时，网络泛化的很好；在其他情况下，预测的测试位姿都是更加解决具有相似外观的训练图像中的位姿。场景下，训练得到的平移基不能准确的对测试位置进行建模，这表明了需要更多的训练数据。
​		（3）当测试图像与训练图像有较少的视觉重合时，基于结构的Active Search方法失败或提供了不足够准确的解，由于没有足够的匹配对。

4. **基于图像检索的方法**

   设 $\mathcal{I}$为测试图像， $\mathcal{J}$ 为与测试图像有很大相似性的训练图像，用数学定义如下：$\alpha^{\mathcal{I}}$ as $\alpha^{\mathcal{I}}=\alpha^{\mathcal{J}}+\Delta^{\mathcal{I}}$ 。

运用上文中的公式（3）可以得到如下测试图像的位姿：

​									$$\left(\begin{array}{l}\hat{\mathbf{c}}_{\mathcal{I}} \\ \hat{\mathbf{r}}_{\mathcal{I}}\end{array}\right)=\left(\begin{array}{l}\hat{\mathbf{c}}_{\mathcal{J}} \\ \hat{\mathbf{r}}_{\mathcal{J}}\end{array}\right)+\left(\begin{array}{c}\sum_{j=1}^{n} \Delta_{j}^{\mathcal{T}} \mathbf{c}_{j} \\ \sum_{j=1}^{n} \Delta_{j}^{T} \mathbf{r}_{j}\end{array}\right)=\left(\begin{array}{c}\hat{\mathbf{c}}_{\mathcal{J}} \\ \hat{\mathbf{r}}_{\mathcal{J}}\end{array}\right)+\left(\begin{array}{c}\hat{\mathbf{c}}_{\mathcal{I}, \mathcal{J}} \\ \hat{\mathbf{r}}_{\mathcal{I}, \mathcal{J}}\end{array}\right)$$  （4）

式（4）里， $\left(\hat{\mathbf{c}}_{\mathcal{I}, \mathcal{J}}, \hat{\mathbf{r}}_{\mathcal{I}, \mathcal{J}}\right)$ 是pose offset。

​		式（4）中已经体现出了图像检索与APR之间有很强的相关性。对于一副测试图像的hidden feature，可以将其分解成与其最相关的训练图像的hidden feature和offset。这个过程的相似性寻找是网络给出的，而作者认为可以利用图像检索的方式找到最相似图像，如通过Bag-of-Words的方法，而且也可以通过那些比较相似的图像作offset，因此作者有如下定义：

​						 $\sum_{i=1}^{k} a_{i}\left(\mathbf{c}_{\mathcal{J}_{i}}, \mathbf{r}_{\mathcal{J}_{i}}\right), \sum a_{i}=1,$ 

上述表明top- $k$个被检索到的相似图像，对他们的pose按照相似度进行线性加权。如何得到$a_{i}$呢？作者具体的做法是先计算各测试图像的描述子 $\mathbf{d}(\mathcal{I}),$ 然后最小化 $\left\|\mathbf{d}(\mathcal{I})-\sum_{i=1}^{k} a_{i} \mathbf{d}\left(\mathcal{J}_{i}\right)\right\|_{2}$ subject to $\sum a_{i}=1 .$ 但注意这里的base poses认为是训练图像的pose，而基于CNN回归的方法的base pose是网络学习而得。

​		作者之后还提出了APR方法与RPR（relative pose regression）方法之间的联系，以我个人做过VO方面的经验来看，VO的训练过程应该是一种RPR的训练过程，输入通常是两幅图像，然后输出两者之间的相对位姿。我按照作者的理论来解释，CNN将两幅图像映射到hidden feature，或者说映射出来的是两者的offset，然后全连接层的权重依然是base pose。如果base pose泛化能力不够，那么效果仍然不好。这就是为什么如果只拿前后顺序输入图像训练网络无法预测后前顺序输入图像的pose，因为base pose没有这方面的信息。

​		文中作者对APR方法的总结如下：位姿回归的方法依赖大量训练数据，即便训练数据足够多，但仍与基于传统几何法相比效果会差一些。APR方法距离实用还有很长的路要走。





> [1]https://blog.csdn.net/wumo1556/article/details/88957142
>
> [2]https://www.bilibili.com/video/av65675814/
>
> [3]论文链接：https://arxiv.org/abs/1903.07504v1
>
> [4]https://arxiv.org/abs/1704.07813


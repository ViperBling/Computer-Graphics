[TOC]

# Lecture01. Overview of Computer Graphics

画面的明亮程度可以成为评判游戏画面好坏的标准。特效最难的点在于经常看到的东西。主要讲的内容：

- 光栅化
- 几何形体
- 光线追踪
- 动画和模拟

![image-20200527173522234](GAMES101_Note.assets/image-20200527173522234.png)

不包括API、建模、引擎使用、CV等。CV和图形学的区别：

![image-20200527173839254](GAMES101_Note.assets/image-20200527173839254.png)

# Lecture02. Review of Linear Algebra

图形学依赖于：

![image-20200527180256017](GAMES101_Note.assets/image-20200527180256017.png)

- 基础数学：线性代数、微积分、统计
- 基础物理：光学、机械
- 其他：信号处理、数值分析
- 美学

## 叉乘

叉乘可以判定左和右，还可以判定内与外。

![image-20200527182941346](GAMES101_Note.assets/image-20200527182941346.png)

左图中，可以通过对向量$\vec a$和$\vec b$做叉乘，判断$\vec b$在$\vec a$的左侧还是右侧，如果叉乘结果是正向量，说明在左侧。

## 矩阵

![image-20200529151806157](GAMES101_Note.assets/image-20200529151806157.png)

# Lecture03. Transformation

## Why study transformation

### 1. Modeling 模型变换

移动相机视野中的3D模型的变换；模型移动过程中的旋转变换；模型的缩放变换

### 2. Viewing 视图变换

从3D模型转换成2D的图片

![image-20200529153429097](GAMES101_Note.assets/image-20200529153429097.png)

## 2D transformation



缩放变换：

![image-20200529153634703](GAMES101_Note.assets/image-20200529153634703.png)

镜像变换：

![image-20200529153737151](GAMES101_Note.assets/image-20200529153737151.png)

切变：

![image-20200529154000362](GAMES101_Note.assets/image-20200529154000362.png)

底边不动，$y=1$的位置移动$a$。

旋转变换：

![image-20200529154531563](GAMES101_Note.assets/image-20200529154531563.png)

令$(x,y)=(1,0),(0,1)$带入变换即可得到旋转公式。旋转矩阵$R_{\theta}$是正交阵

上述变换都可以写成矩阵乘向量的形式：

![image-20200603161609851](GAMES101_Note.assets/image-20200603161609851.png)

这种变换就线性变换。

## Homogeneous coordinates 齐次坐标系

上面的变换形式无法套用到平移变换中：

![image-20200603162104847](GAMES101_Note.assets/image-20200603162104847.png)

![image-20200603162111972](GAMES101_Note.assets/image-20200603162111972.png)

为此我们要引入齐次坐标系。将原来2维的向量扩充一维，加入齐次项，对二维的点加入一个齐次项1，二维的向量加入齐次项0：

![image-20200603162711220](GAMES101_Note.assets/image-20200603162711220.png)

这样平移变换可以写成矩阵形式：

![image-20200603162204517](GAMES101_Note.assets/image-20200603162204517.png)

为什么点增加的是1，向量增加的是0呢？因为向量具有平移不变性，它表示的是一个方向，所以平移不应该改变它的指向。更进一步有：

![image-20200603163121382](GAMES101_Note.assets/image-20200603163121382.png)

齐次坐标中规定，只要第三个齐次项$\omega$不为0，都可以化成标准形式。其中point+point的结果还是一个点，表示的是两点的中点。

### 1. Affine Transformation

![image-20200603163703536](GAMES101_Note.assets/image-20200603163703536.png)

在表示仿射变换的时候，变换矩阵的最后一行始终是001.

### 2. Inverse Transform

变换矩阵的逆矩阵就是逆变换。

### 3. Composing Transforms

多个变换叠加就是多个矩阵相乘：

![image-20200603164424946](GAMES101_Note.assets/image-20200603164424946.png)

最后的变换在最左边。

### 4. Decomposing Complex Transforms

可以将矩阵的变换进行分解：

![image-20200603164931308](GAMES101_Note.assets/image-20200603164931308.png)

以上图为例，想要实现以$c$点为中心的旋转，我们可以将图片整个沿$-c$的方向移动到原点，采用变换矩阵$T(-c)$，然后旋转一个角度，再沿着$c$方向移动回去即可。

## 3D Transformation

类似二维变换，三维情况再加一个齐次坐标即可：

![image-20200603165141990](GAMES101_Note.assets/image-20200603165141990.png)

# Lecture04. Transformation Cont.

## 3D Transformations

绕轴旋转：

![image-20200603170619228](GAMES101_Note.assets/image-20200603170619228.png)

为什么Y轴是转置的。因为按照叉乘的规律，$z\times x=y$，而不是$x \times z$。

任意三维空间的旋转可以写成绕三个坐标轴旋转的组合：

![image-20200603171239112](GAMES101_Note.assets/image-20200603171239112.png)

罗德里格斯旋转公式：

![image-20200603171454967](GAMES101_Note.assets/image-20200603171454967.png)

## Viewing Transformation(观测变换)MVP

### 1. View(视图)/Camera transformation

首先定义相机，包括位置、朝向、上方向。向上方向为了明确相机的旋转（俯仰角度）：

![image-20200603172351575](GAMES101_Note.assets/image-20200603172351575.png)

我们约定俗成固定相机在原点，沿着-z方向看，上方向是y轴正方向。在设定场景的时候需要把相机放到约定俗成的位置，这里涉及到一个变换$M_{view}$：

![image-20200603173203123](GAMES101_Note.assets/image-20200603173203123.png)

变换过程如下：

![image-20200603173602789](GAMES101_Note.assets/image-20200603173602789.png)

难点在求旋转矩阵上，直接将某个轴旋转到标准轴不好求，可以先求逆，让标准轴旋转到当前轴的位置。然后求逆矩阵。

### 2. Projection(投影) transformation

正交投影不会近大远小，透视投影则会。

![image-20200603174926881](GAMES101_Note.assets/image-20200603174926881.png)

#### Orthographic projection（正交投影）

将摄像机摆到指定位置后，将Z轴去掉，就得到了正交投影：

![image-20200603175545654](GAMES101_Note.assets/image-20200603175545654.png)



#### Perspective projection（透视投影）
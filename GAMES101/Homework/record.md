[TOC]

# Assignment00. 旋转点

此次作业采用Eigen线性代数运算库进行空间中点和向量的运算。

最基础的向量表示：`Eigen::Vector3f v(1.0f, 2.0f, 3.0f)`，向量之间加减数乘按照普通规则来，矩阵表示如下：

```c++
// Example of matrix
std::cout << "Example of matrix \n";
// matrix definition
Eigen::Matrix3f i,j;
i << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
j << 2.0, 3.0, 1.0, 4.0, 6.0, 5.0, 9.0, 7.0, 8.0;
```

`Matrix3f`表示一个$3\times 3$的矩阵。

作业内容是：

给定一个点 $P=(2,1)$，将该点绕原点先逆时针旋转$45^\circ$，再平移$ (1,2)$, 计算出 变换后点的坐标（要求用齐次坐标进行计算）。

首先要声明一个三维向量，包含一维的齐次坐标。按照课程中的方法，旋转变换和平移变换可以用矩阵的乘积表示，如下：
$$
R = R_t*R_\theta = 
\begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 0
\end{bmatrix}

\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$
所以只要计算出$R$，并左乘向量$P=(2,1)$即可：
$$
R = 
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 2 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\frac{\sqrt 2}{2} & -\frac{\sqrt 2}{2} & 0 \\
\frac{\sqrt 2}{2} & \frac{\sqrt 2}{2} & 0 \\
0 & 0 & 1
\end{bmatrix}=
\begin{bmatrix}
\frac{\sqrt 2}{2} & -\frac{\sqrt 2}{2} & 1 \\
\frac{\sqrt 2}{2} & \frac{\sqrt 2}{2} & 2 \\
0 & 0 & 1
\end{bmatrix}
$$
写出程序：

```c++
double torad(const float a) {
    return a * M_PI / 180;
}

int main(){
    Eigen::Vector3f P(2, 1, 1);
    Eigen::Matrix3f R;
	// 需要将角度转换成弧度才能用C++自带的库函数，此外C++中用M_PI表示PI
    R << cos(torad(45)), -sin(torad(45)), 1,
         sin(torad(45)),  cos(torad(45)), 2,
         0,              0,               1;
    cout << "Rotation and Translation of P(2,1): " << endl;
    cout << R * P << endl;

    return 0;
}
```

![image-20200723165759087](record.assets/image-20200723165759087.png)

也可以用Eigen自带的库函数，Eigen自带了旋转函数，在Sapce Transformation中可以查到：

```c++
int main(){
    Vector3f P(2, 1, 1);

    Rotation2Df r(M_PI / 4);	// 构造2维旋转矩阵
    Translation2f t(1, 2);		// 构造平移向量

    Matrix3f rot;				// 最终生成的变换矩阵
    rot.setIdentity();			// 初始化为单位阵
    // block<NRows, NCols>(startRow, startCol);
    rot.block<2, 2>(0, 0) = r.toRotationMatrix();		// 填充旋转矩阵
    rot.block<2, 1>(0, 2) = t.translation();			// 填充平移矩阵

    cout << rot * P << endl;
    
    return 0;
}
```

![image-20200723195633825](record.assets/image-20200723195633825.png)

输出是一样的。这里要注意的就是各类矩阵的类型，一般来说使用浮点型，也就是后缀是f，类型不匹配会报错。Eigen中的旋转有角轴旋转（3D），四元数旋转（3D），旋转矩阵（2D&3D）。这里使用的旋转矩阵方法。具体参见链接：[Eigen使用笔记](http://zhaoxuhui.top/blog/2019/09/03/eigen-note-4.html#12d旋转)

# Assignment01. 旋转与投影

填写一个旋转矩阵和透视投影矩阵，实现对图像的旋转和透视操作。复习一下MVP变换。

https://blog.csdn.net/junzia/article/details/85939783

## Model矩阵

Model矩阵用于模型在模型空间中的转换（一般模型空间是右手系），是无限大的一个空间。Model矩阵隐含了旋转、平移、缩放三种变换。旋转包含三个方向的旋转：
$$
\begin{align}
&\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & \cos\alpha & -\sin\alpha & 0\\
0 & \sin\alpha & \cos\alpha & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \quad 绕X轴旋转 \\ \\
&\begin{bmatrix}
\cos\beta & 0 & \sin\beta & 0\\
0 & 1 & 0 & 0\\
-\sin\beta & 0 & \cos\beta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \quad 绕Y轴旋转 \\ \\
&\begin{bmatrix}
\cos\gamma & -\sin\gamma & 0 & 0\\
\sin\gamma & \cos\gamma & 0 & 0\\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \quad 绕Z轴旋转
\end{align}
$$

## View矩阵

View矩阵是观察者方向相对物体位置变化时用于描述对应变化的矩阵。观察者相对物体的位置变化也可以采用物体自身的变化来描述，可以用Model矩阵来描述，因为运动是相对的。

View矩阵的作用是将模型从模型世界的坐标系中转换到相对相机的观察坐标系中。相机的位置、观察的方向和相机向上的向量共同构成了观察坐标系，View矩阵就是将物体从模型坐标系变换到观察坐标系。

从世界坐标变换到相机空间坐标，一般有两步：

- 首先将世界坐标的基旋转到和相机空间基重合的位置，实际上就是从一个向量空间到另一个向量空间的变换，求出过渡矩阵即可：
  $$
  R = \begin{bmatrix}
  u_x & u_y & u_z & 0 \\
  v_x & v_y & v_z & 0 \\
  w_x & w_y & w_z & 0 \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  $$
  这就是旋转矩阵。

- 然后将世界坐标平移到相机坐标的原点，由此可以得到旋转矩阵：
  $$
  T = \begin{bmatrix}
  1 & 0 & 0 & -t_x \\
  0 & 1 & 0 & -t_y \\
  0 & 0 & 1 & -t_z \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  $$
  

## Projection矩阵

投影矩阵用于将相机空间转换到裁剪空间。因为屏幕不是无限大的。包括正交投影和透视投影。

### 正交投影

正交投影中$x,y$轴坐标不变，直接将$z$轴坐标去除。假设点$P(x,y,z)$经过model和view变换后得到$P_1(x_1, y_1, z_1)$，$P_1$经过正交投影变换得到$P_2(x_2,y_2,z_2)$，点$P_2$的$x,y,z$分量都在$[-1,1]$之间，因为正交投影变换是将点映射到标准视体（中心在原点，边长为2的立方体）中。

正交投影矩阵有六个参数，即待投影空间的范围，上平面$t$，下平面$b$，左平面$l$，右平面$r$，近平面$n$，远平面$f$。投影的过程是将待投影空间变换到标准视体的过程，标准视体是$[-1,1]\times [-1,1]\times [-1,1]$的一个范围，所以变换的过程就包括平移和缩放两个过程。包含两个矩阵：
$$
\begin{bmatrix}
x_{can} \\ y_{can} \\ z_{can} \\ 1 
\end{bmatrix}=\begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & 0 \\
0 & \frac{2}{t-b} & 0 & 0 \\
0 & 0 & \frac{2}{n-f} & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
1 & 0 & 0 & -\frac{l+r}2 \\
0 & 1 & 0 & -\frac{t+b}2 \\
1 & 0 & 1 & -\frac{n+f}2 \\
0 & 0 & 0 & 1
\end{bmatrix} \begin{bmatrix} x \\ y \\ z \\1 \end{bmatrix}
 = M_oP
$$

### 透视投影

透视投影的作用就是将一个视锥（符和近大远小）转换成一个标准视体，需要先转换成正交投影，然后再变换到标准视体。输入参数有4个，广角fov（fovx或fovy）、宽高比aspect，近平面n，远平面f。采用这种方式定义时：

![image-20200804155601469](record.assets/image-20200804155601469.png)

如图，可以得到透视投影转换成正交投影后，正交投影的范围$l,r,t,b,n,f$之间的关系：
$$
-l = r, aspect = \frac rt\\ -b=t, \tan\frac{fovY}{2} = \frac t{|n|} \\
$$
其中$n,f$已知，所以我们就能根据输入参数构建$M_o$，然后乘以透视矩阵$M_p$。$M_p$的过程是将视锥压缩成平行六面体（就是正交投影前的那个）。注意到视锥中物体的$y$坐标是和$z$成比例的：

![image-20200804165204547](record.assets/image-20200804165204547.png)

同理对于$x$有$x'=\frac nz x$。现在我们知道了$x,y$如何变换，也就是坐标$(x,y,z,1)$怎么样从视锥中变到立方体：

![image-20200804170037172](record.assets/image-20200804170037172.png)
$$
M_p\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} =
\begin{bmatrix}
nx/z \\ ny/z \\ ? \\ 1
\end{bmatrix} = 
\begin{bmatrix}
nx \\ ny \\ ? \\ z
\end{bmatrix}
$$
现在未知的是$z$如何变化，根据上面的结果可以构造一个矩阵如下：
$$
M_p = \begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
? & ? & ? & ? \\
0 & 0 & 1 & 0 
\end{bmatrix}
$$
第三行是未知的，可以带入特殊点进行验证。首先在近平面的所有点的坐标是不会发生变化的，即$M_pP_n = P_n$，也就是当$z=n$时，坐标点不发生变化。此外远平面上所有点的$z$坐标也不会发生变化，即$z = f$时，坐标点的第三个分量不变。

首先，当$z=n$时，点$(x,y,n,1)$经过变换还是$(x,y,n,1)$，同乘以$n$，得到$(nx,ny,n^2,n)$，假设：
$$
M_p = \begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
a & b & c & d \\
0 & 0 & 1 & 0
\end{bmatrix}
$$
对比变换结果，因为$n^2 = ax+by+cn+d$，可得$a = b = 0$。

然后根据第二点，远平面上$z$坐标不变，$(x,y,f,1)$经过变换后为$(nx,ny,f,1)$，同乘以$f$，得到$(nfx,nfy,f^2,f)$，即$cf +d = f^2$，结合上面得式子：
$$
\begin{cases}
cn+d=n^2 \\
cf+d=f^2
\end{cases}
$$
求得$c=n+f,d=-nf$，即：
$$
M_p = \begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n+f & -nf \\
0 & 0 & 1 & 0
\end{bmatrix}
$$
有之前的结论得到：
$$
M_o = \begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & -\frac{l+r}2 \\
0 & \frac{2}{t-b} & 0 & -\frac{t+b}2 \\
0 & 0 & \frac{2}{n-f} & -\frac{n+f}2 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$
根据fovY，aspect和六面体的关系得：
$$
-b = t = \tan\frac{fovY}2\times |n| \\
-l = r = aspect \times t
$$
带入$M_o$就得到了完整的投影矩阵。

## 视口变换

将上面变换后的规范化设备坐标系NDC光栅化到2D平面进行显示。

作业要求实现旋转和透视投影矩阵，首先实现绕Z轴旋转的变换矩阵，函数接受一个角度参数，然后返回旋转矩阵：

```c++
Eigen::Matrix4f get_model_matrix(float rotation_angle) {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    
    // Rotation2Df rot(rotation_angle);
    Eigen::Matrix2f rot;

    float rad = rotation_angle * MY_PI / 180;
    rot << cos(rad), -sin(rad),  
           sin(rad), cos(rad);
	// 注释掉的是使用自带的旋转类的方法。
    // model.block<2, 2>(0, 0) = rot.toRotationMatrix();
    model.block<2, 2>(0, 0) = rot;

    return model;
}
```

下面实现透视投影矩阵，根据上面的描述，首先得到正交投影阵$M_o$：

```c++
Eigen::Matrix4f get_orthogonality_matrix(float xLeft, float xRight, float yTop,
                                         float yBottom, float zNear, float zFar)
{
    Eigen::Matrix4f ortho = Eigen::Matrix4f::Identity();
	// 分成缩放和平移两个变换
    Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity()；
	// 缩放矩阵
    scale << 2 / (xRight - xLeft), 0,                    0,                  0,
             0,                    2 / (yTop - yBottom), 0,                  0,
             0,                    0,                    2 / (zNear - zFar), 0,
             0,                    0,                    0,                  1;
    trans << 1, 0, 0, -(xLeft + xRight) / 2,
             0, 1, 0, -(yTop + yBottom) / 2,
             0, 0, 1, -(zNear + zFar) / 2,
             0, 0, 0, 1;
    ortho = scale * trans;

    return ortho;
} 
```

然后构造透视投影矩阵，包括前面的正交投影矩阵以及将视锥变成平行六面体的变换：

```c++
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    
    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    // 注意这里zNear一定为负，因为以观察点为原点，向z轴负方向看去，远近平面都是负的。否则得到的三角形头向下。
    zNear = zNear > 0 ? -zNear : zNear;
    // 计算视锥的边界
    float yTop = tanf(eye_fov / 2)* abs(zNear);
    float yBottom = -yTop;

    float xRight = aspect_ratio * yTop;
    float xLeft = -xRight;
	// 构造M_P矩阵
    projection(0, 0) = zNear;
    projection(1, 1) = zNear;
    projection(2, 2) = zNear + zFar;
    projection(2, 3) = -zNear * zFar;
    projection(3, 2) = 1;
    projection(3, 3) = 0;
	// 正交投影变换矩阵 * M_P就得到了透视投影矩阵
    projection =  get_orthogonality_matrix(xLeft, xRight, yTop, yBottom, zNear, zFar) * projection;

    return projection;
}
```

![image-20200804184155653](record.assets/image-20200804184155653.png)

结果如上图所示，按”A,D“可以进行旋转。

后面还有个提高项，构造一个函数，得到绕任意过原点的轴的旋转矩阵。旋转矩阵的推导可以由原向量$v$，轴向量$n$，旋转后的向量$v'$，来确定，具体关系如下：

![image-20200805164318114](record.assets/image-20200805164318114.png)

得到的矩阵如下，加上齐次坐标：
$$
R(\boldsymbol n, \theta) =\begin{bmatrix}
n_x^2(1-\cos\theta)+\cos\theta & n_xn_y(1-\cos\theta)+n_z\sin\theta & n_xn_z(1-\cos\theta)-n_y\sin\theta & 0 \\
n_xn_y(1-\cos\theta)-n_z\sin\theta & n_y^2(1-\cos\theta)+\cos\theta & n_yn_z(1-\cos\theta)+n_x\sin\theta & 0 \\
n_xn_z(1-\cos\theta)+n_y\sin\theta & n_yn_z(1-\cos\theta)-n_x\sin\theta & n_z^2(1-\cos\theta)+\cos\theta & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$
代码如下：

```c++
Eigen::Matrix4f get_rotation(Vector3f axis, float angle) {
    Matrix4f rot = Matrix4f::Identity();

    float rad = angle * MY_PI / 180;
    // 轴向量的三个分量
    float n_x = axis(0);
    float n_y = axis(1);
    float n_z = axis(2);

    rot << n_x * n_x * (1 - cos(rad)) + cos(rad), n_x * n_y * (1 - cos(rad)) + n_z * sin(rad), n_x * n_z * (1 - cos(rad)) - n_y * sin(rad), 0,
           n_x * n_y * (1 - cos(rad)) - n_z * sin(rad), n_y * n_y * (1 - cos(rad)) + cos(rad), n_y * n_z * (1 - cos(rad)) + n_x * sin(rad), 0,
           n_x * n_z * (1 - cos(rad)) + n_y * sin(rad), n_y * n_z * (1 - cos(rad)) - n_x * sin(rad), n_z * n_z * (1 - cos(rad)) + cos(rad), 0,
           0, 0, 0, 1;   

    return rot;
}
```

# Assignments02. 三角形绘制和Z-Buffering

绘制一个实心的三角形，也就是栅格化一个三角形。同时处理深度值，显示两个三角形时，正确处理重叠关系。

要完成两个函数，第一个`rasterize_triangle(const Triangle& t)`，内部流程如下：

- 创建三角形的二维Bounding Box
- 遍历Bounding Box中所有的像素，然后判断像素的中心点是否在三角形内部（像素使用整数索引）
- 如果在内部，将其位置出的插值深度值与深度缓冲区中的相应值比较
- 如果当前点更靠近相机，请设置像素颜色并更新深度缓冲区

第二个`static bool insideTriangle(int x, int y, const Vector3f* _v)`用来判断像素是否在三角形内部。我们只知道三角形三个顶点的深度值，内部的深度值需要通过插值得到。

## 判断点在三角形内

首先实现判断点是否在三角形内部的函数，函数接受三个参数，点$P(x,y)$，和一个指向三角形三个顶点的数组，里面包含三个顶点`_v[0], _v[1], _v[2]`。判断依据是叉乘，同时点的顺序是按照逆时针排列的。按照$v[0]\rightarrow v[1],v[1]\rightarrow v[2],v[2]\rightarrow v[1]$的顺序组成三角形三条边，

分析一下源码：

```c++
//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();   
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}
```

光栅化三角形函数接收一个三角形类，然后经过`toVector4()`函数调用，它的过程如下：

```c++
std::array<Vector4f, 3> Triangle::toVector4() const
{
    std::array<Eigen::Vector4f, 3> res;
    std::transform(std::begin(v), std::end(v), res.begin(), [](auto& vec) { return Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1.f); });
    return res;
}
```

`toVector4()`将三角形的三个顶点转换成齐次坐标的形式，存储在一个长度为3的数组`res`中。所以我们上面返回的`v`就是一个`array<Vector4f, 3>`类型的数组。在调用`insideTriangle`时，第三个参数是一个指向`Vector3f`的指针，仿照注释中的调用方法，直接调用`t.v`即可得到三角形三个顶点所在位置。然后根据三个顶点计算三条边的向量（顺时针方向）：

```c++
Vector3f ab = _v[1] - _v[0];
Vector3f bc = _v[2] - _v[1];
Vector3f ca = _v[0] - _v[2];
```

`_v[i]`都是包含3个坐标的齐次坐标，叉乘是三维空间向量，所以不能用`Vector2f`，必须是`Vector3f`，所以直接两点坐标相减即可。然后同样的方法得到待定点和三角形顶点的向量。判断点是否在三角形内部的方法：
$$
cross1 = ab\times pa \\cross2= bc\times pb\\ cross3 = ca\times pc
$$
然后三个叉乘结果的$z$方向在同一个方向时就说明点在三角形内，否则在三角形外。完整代码如下：

```c++
static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f point = {x + 0.5, y + 0.5, 1.f};

    Vector3f ab = _v[1] - _v[0];
    Vector3f bc = _v[2] - _v[1];
    Vector3f ca = _v[0] - _v[2];

    Vector3f pa = _v[0] - point;
    Vector3f pb = _v[1] - point;
    Vector3f pc = _v[2] - point;

    Vector3f cross1 = ab.cross(pa);
    Vector3f cross2 = bc.cross(pb);
    Vector3f cross3 = ca.cross(pc);

    return cross1.z() * cross2.z() > 0 && cross2.z() * cross3.z() > 0 && cross3.z() * cross1.z() > 0;
}
```

## 光栅化三角形

光栅化三角形的过程就是扫描Bounding Box，判断每个点是否在三角形内部，如果在，根据三角形顶点插值确定颜色，然后更新z-buffer。如果不在，那么不改变像素颜色。

根据提示，首先确定Bounding Box，上边界是三角形上顶点的纵坐标，下边界是三角形下顶点的纵坐标，左右边界类似。Bounding Box采用4个值确定，分别代表左右$x$，上下$y$。

```c++
std::array<float, 4> bbx{0, 0, 0, 0}; // {x1, y1, x2, y2}
for (auto point : v) {
    bbx[0] = bbx[0] < point.x() ? bbx[0] : point.x();
    bbx[1] = bbx[1] < point.y() ? bbx[1] : point.y();
    bbx[2] = bbx[2] > point.x() ? bbx[2] : point.x();
    bbx[3] = bbx[3] > point.y() ? bbx[3] : point.y();
}
```

然后在Bounding Box的范围内，对所有的像素进行遍历决定像素颜色。框架中给出了设置像素颜色的函数`void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)`，根据点的位置设置对应的颜色。颜色的获取函数是三角形类的接口`getColor()`，获取顶点颜色，然后输出：

```c++
for (int i = std::floor(bbx[0]); i < std::ceil(bbx[2]); ++i) {
    for (int j = std::floor(bbx[1]); j < std::ceil(bbx[3]); ++j) {
        if (insideTriangle(i, j, t.v)) {
            Vector3f point = {i, j, 1.f};
            set_pixel(point, t.getColor());
        }
    }
}
```

但是这个过程没有加入Z-Buffer。回顾一下深度缓冲算法，对每个像素点存储一个深度值，然后根据要绘制的三角形的$z$坐标，判断其与当前深度值的大小，选择较小的显示。三角形内部的深度值是通过顶点插值得到的：

```c++
// 计算三角形重心坐标
auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
// v.w()是齐次坐标值
float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
// 得到深度值
z_interpolated *= w_reciprocal;
```

然后根据得到的深度值`z_interpolated`来更新`depth_buf`，因为`z_interpolated`是大于0的，要取负值进行比较然后绘制像素：

```c++
int idx = get_index(x, y);
if (-z_interpolated < depth_buf[idx]) {
	depth_buf[idx] = -z_interpolated;
// TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
	set_pixel(point, t.getColor());
}
```

完整函数如下：

```c++
//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    std::array<float, 4> bbx{0, 0, 0, 0}; // {x1, y1, x2, y2}

    for (auto point : v) {
        bbx[0] = bbx[0] < point.x() ? bbx[0] : point.x();
        bbx[1] = bbx[1] < point.y() ? bbx[1] : point.y();
        bbx[2] = bbx[2] > point.x() ? bbx[2] : point.x();
        bbx[3] = bbx[3] > point.y() ? bbx[3] : point.y();
    }

    for (int x = std::floor(bbx[0]); x < std::ceil(bbx[2]); ++x) {
        for (int y = std::floor(bbx[1]); y < std::ceil(bbx[3]); ++y) {
            if (insideTriangle(x, y, t.v)) {
                Vector3f point = {x, y, 1.f};
                
                // If so, use the following code to get the interpolated z value.
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                // std::cout << z_interpolated << std::endl;

                int idx = get_index(x, y);
                if (-z_interpolated < depth_buf[idx]) {
                    depth_buf[idx] = -z_interpolated;
                    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                    set_pixel(point, t.getColor());
                }
            }
        }
    }
    
}
```

然后是使用MSAA反走样。将一个像素分割成4个小像素，用这4个小像素颜色的均值来决定像素的颜色。当小像素落入三角形内部的数量越多，像素的颜色就越明显，否则越暗淡。






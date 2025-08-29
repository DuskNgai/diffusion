# Mean Flow

Mean Flow 模型基于 Retified Flow，该路径将原始数据 $\mathbf{x}_0$ 线性插值到纯噪声 $\boldsymbol{\epsilon}$：
$$
\mathbf{x}_{t} = (1 - t) \mathbf{x}_{0} + t \boldsymbol{\epsilon}, \quad t \in [0, 1]
$$
该路径的瞬时速度（即对参数 $t$ 的导数）是恒定的：
$$
\mathbf{v}(\mathbf{x}_{t}, t) = \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \boldsymbol{\epsilon} - \mathbf{x}_0
$$

## Average Velocity

我们定义在时刻 $s$ 和 $t$ 之间的平均速度为瞬时速度在该时间段内积分的平均值：
$$
\mathbf{u}(\mathbf{x}_{t}, s, t) := \frac{1}{t - s} \int_{s}^{t} \mathbf{v}(\mathbf{x}_{\tau}, \tau) \mathrm{d}\tau
$$

根据其积分定义，平均速度 $\mathbf{u}$ 天然满足一个重要的线性组合性质：
$$
(t - s) \mathbf{u}(\mathbf{x}_{t}, s, t) = (t - r) \mathbf{u}(\mathbf{x}_{t}, r, t) + (r - s) \mathbf{u}(\mathbf{x}_{r}, s, r)
$$
其中 $r$ 是 $t$ 和 $s$ 之间的任意时刻。

这一性质的核心优势在于：如果我们能训练一个网络 $u_{\theta}(\mathbf{x}_{t}, s, t)$ 来近似这个平均速度，我们就可以通过评估 $u_{\theta}(\mathbf{x}_{1}, 0, 1)$ 来实现单步生成。由于 $\mathbf{x}_1 = \boldsymbol{\epsilon}$，这意味着我们可以从一个纯噪声输入一步解码出原始数据。此外，我们还有：
$$
\mathbf{v}(\mathbf{x}_{t}, t) = \lim_{s \to t} \frac{1}{t - s} \int_{s}^{t} \mathbf{v}(\mathbf{x}_{\tau}, \tau) \mathrm{d}\tau = \mathbf{u}(\mathbf{x}_{t}, t, t)
$$

## Mean Flow Identity

直接通过积分来计算平均速度 $\mathbf{u}$ 是开销大。我们可以通过一个微分恒等式来构建一个可行的训练目标。考虑对平均速度的积分形式进行求导：
$$
\begin{aligned}
(t - s) \mathbf{u}(\mathbf{x}_{t}, s, t) &= \int_{s}^{t} \mathbf{v}(\mathbf{x}_{\tau}, \tau) \mathrm{d}\tau \\
\frac{\mathrm{d}}{\mathrm{d} t} [(t - s) \mathbf{u}(\mathbf{x}_{t}, s, t)] &= \frac{\mathrm{d}}{\mathrm{d} t} \left[\int_{s}^{t} \mathbf{v}(\mathbf{x}_{\tau}, \tau) \mathrm{d}\tau\right] \\
\mathbf{u}(\mathbf{x}_{t}, s, t) + (t - s) \frac{\mathrm{d}}{\mathrm{d} t} \mathbf{u}(\mathbf{x}_{t}, s, t) &= \mathbf{v}(\mathbf{x}_{t}, t) \\
\mathbf{u}(\mathbf{x}_{t}, s, t) &= \mathbf{v}(\mathbf{x}_{t}, t) - (t - s) \frac{\mathrm{d}}{\mathrm{d} t} \mathbf{u}(\mathbf{x}_{t}, s, t)
\end{aligned}
$$
这个恒等式建立了瞬时速度 $\mathbf{v}$ 和平均速度 $\mathbf{u}$ 之间的直接关系。

其中，$\mathbf{u}(\mathbf{x}_{t}, s, t)$ 的全导数 $\dfrac{\mathrm{d}}{\mathrm{d} t} \mathbf{u}$ 可以通过链式法则展开：
$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t} \mathbf{u}(\mathbf{x}_{t}, s, t) &= \frac{\partial \mathbf{u}}{\partial \mathbf{x}_{t}} \frac{\mathrm{d} \mathbf{x}_{t}}{\mathrm{d} t} + \frac{\partial \mathbf{u}}{\partial s} \frac{\mathrm{d} s}{\mathrm{d} t} + \frac{\partial \mathbf{u}}{\partial t} \frac{\mathrm{d} t}{\mathrm{d} t} \\
&= \frac{\partial \mathbf{u}}{\partial \mathbf{x}_{t}} \mathbf{v}(\mathbf{x}_{t}, t) + \dfrac{\partial \mathbf{u}}{\partial s} \cdot 0 + \frac{\partial \mathbf{u}}{\partial t} \cdot 1 \\
&=
\begin{bmatrix}
\dfrac{\partial \mathbf{u}}{\partial \mathbf{x}_{t}} & \dfrac{\partial \mathbf{u}}{\partial s} & \dfrac{\partial \mathbf{u}}{\partial t}
\end{bmatrix}
\begin{bmatrix}
\mathbf{v} \\ 0 \\ 1
\end{bmatrix}
\end{aligned}
$$
这个计算可以高效地通过雅可比向量积 (JVP) 实现。

## 算法实现

```python
def training():
    x0 = sample_data()
    epsilon = sample_noise()
    s, t = sample_timestep()

    xt = (1 - t) * x0 + t * epsilon
    vt = epsilon - x0

    u, dudt = jvp(model, (xt, s, t), (vt, 0, 1))
    u_tgt = vt - (t - s) * dudt

    loss = loss_fn(u, u_tgt.detach())

def sampling():
    epsilon = sample_noise()
    x0 = epsilon - model(epsilon, 0, 1)
```

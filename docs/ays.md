# Align Your Steps Re-formulate

## Related

### Reverse SDEs

$$
\mathrm{d}\mathbf{x}_{t} = \left[\frac{\dot{s}(t)}{s(t)} \mathbf{x}_{t} - 2 \left(\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}\right)\sigma(t)^{2} \nabla_{\mathbf{x}_{t}} \log p_{t}(\mathbf{x}_{t})\right] \mathrm{d}t + \sigma(t)\sqrt{2 \left(\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}\right)} \mathrm{d}\mathbf{w}_{t}
$$

### Case Study

令 $\mathbb{P}_{\text{data}}(\mathbf{x}_{0}) = \mathcal{N}(\mathbf{0}, c^{2}\mathbf{I})$。

$\mathbf{x}_{t}$ 分布的表达式：
$$
p_{t}(\mathbf{x}_{t}) = \mathcal{N}\left(\mathbf{0}, \left[s(t)^{2}c^{2} + \sigma(t)^{2}\right]\mathbf{I}\right)
$$
分数的表达式：
$$
\nabla_{\mathbf{x}_{t}} \log p_{t}(\mathbf{x}_{t}) = -\frac{\mathbf{x}_{t}}{s(t)^{2}c^{2} + \sigma(t)^{2}}
$$

## $x_{0}$-prediction

$$
\nabla_{\mathbf{x}_{t}} \log p_{t}(\mathbf{x}_{t}) = \frac{s(t)\hat{x}_{\theta}(\mathbf{x}_{t}, s(t), \sigma(t)) - \mathbf{x}_{t}}{\sigma(t)^{2}}
$$
带入反向 SDE，得到：
$$
\mathrm{d}\mathbf{x}_{t} = \left\{ \left(2\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}\right)\mathbf{x}_{t} - 2\left(\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}\right)s(t)\hat{x}_{\theta}(\mathbf{x}_{t}; s(t), \sigma(t)) \right\} \mathrm{d}t + \sigma(t)\sqrt{2\left(\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}\right)} \mathrm{d}\mathbf{w}_{t}
$$

### Case Study

$\hat{x}_{\theta}$ 的表达式：
$$
\begin{aligned}
\frac{s(t)\hat{x}_{\theta}(\mathbf{x}_{t}, s(t), \sigma(t)) - \mathbf{x}_{t}}{\sigma(t)^{2}} &= -\frac{\mathbf{x}_{t}}{s(t)^{2}c^{2} + \sigma(t)^{2}} \\
\hat{x}_{\theta}(\mathbf{x}_{t}, s(t), \sigma(t)) &= -\frac{s(t)c^{2}}{s(t)^{2}c^{2} + \sigma(t)^{2}}\mathbf{x}_{t}
\end{aligned}
$$
期望的表达式：
$$
\begin{aligned}
&\quad \ \ \mathbb{E}_{\mathbf{x}_{t} \sim p_{t}, \mathbf{x}_{t_{i}} \sim p_{t_{i} \mid t}} \left\| \frac{s(t)c^{2}}{s(t)^{2}c^{2} + \sigma(t)^{2}}\mathbf{x}_{t} - \frac{s(t_{i})c^{2}}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}\mathbf{x}_{t_{i}} \right\|^{2} \\
&= c^{4} \mathbb{E}_{\mathbf{x}_{t} \sim p_{t}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left\| \frac{s(t)}{s(t)^{2}c^{2} + \sigma(t)^{2}}\mathbf{x}_{t} - \frac{s(t_{i})}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}\left(\frac{s(t_{i})}{s(t)}\mathbf{x}_{t} + \frac{1}{s(t)}\sqrt{s(t)^{2}\sigma(t_{i})^{2} - s(t_{i})^{2}\sigma(t)^{2}}\boldsymbol{\epsilon}\right) \right\|^{2} \\
&\propto \mathbb{E}_{\mathbf{x}_{t} \sim p_{t}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left\| \left(\frac{s(t)}{s(t)^{2}c^{2} + \sigma(t)^{2}} - \frac{s(t_{i})}{s(t)}\frac{s(t_{i})}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}\right) \mathbf{x}_{t} - \frac{s(t_{i})}{s(t)}\frac{\sqrt{s(t)^{2}\sigma(t_{i})^{2} - s(t_{i})^{2}\sigma(t)^{2}}}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}\boldsymbol{\epsilon} \right\|^{2} \\
&= \left(\frac{s(t)}{s(t)^{2}c^{2} + \sigma(t)^{2}} - \frac{s(t_{i})}{s(t)}\frac{s(t_{i})}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}\right)^{2} [s(t)^{2}c^{2} + \sigma(t)^{2}] + \frac{s(t_{i})^{2}}{s(t)^{2}}\frac{s(t)^{2}\sigma(t_{i})^{2} - s(t_{i})^{2}\sigma(t)^{2}}{(s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2})^{2}} \\
&= \frac{s(t)^{2}}{s(t)^{2}c^{2} + \sigma(t)^{2}} - \frac{s(t_{i})^{2}}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}
\end{aligned}
$$
KLUB 的表达式：
$$
\begin{aligned}
\text{KLUB} &\propto \int_{t_{i-1}}^{t_{i}} \left(\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}\right) \frac{s(t)^{2}}{\sigma(t)^{2}} \left[\frac{s(t)^{2}}{s(t)^{2}c^{2} + \sigma(t)^{2}} - \frac{s(t_{i})^{2}}{s(t_{i})^{2}c^{2} + \sigma(t_{i})^{2}}\right] \mathrm{d}t \\
&= \int_{\lambda(t_{i-1})}^{\lambda(t_{i})} \frac{1}{\lambda^{3}} \left[\frac{1}{c^{2} + \lambda^{2}} - \frac{1}{c^{2} + \lambda_{i}^{2}}\right] \mathrm{d}\lambda
\end{aligned}
$$
其中 $\lambda(t) = \sigma(t) / s(t)$，一般是一个单调递增的函数，存在反函数 $t = \lambda^{-1}(\lambda)$。

### General Case

带入 Girsanov 定理，得到：
$$
\begin{aligned}
\text{KLUB} &\propto \int_{t_{i-1}}^{t_{i}} [\frac{\dot{\sigma}(t)}{\sigma(t)} - \frac{\dot{s}(t)}{s(t)}] \frac{s(t)^{2}}{\sigma(t)^{2}}  \mathbb{E}_{\mathbf{x}_{t} \sim p_{t}, \mathbf{x}_{t_{i}} \sim p_{t_{i} \mid t}} \left\| \hat{x}_{\theta}(\mathbf{x}_{t}; s(t), \sigma(t)) - \hat{x}_{\theta}(\mathbf{x}_{t_{i}}; s(t_{i}), \sigma(t_{i})) \right\|^{2} \mathrm{d}t \\
&\propto \int_{\lambda(t_{i-1})}^{\lambda(t_{i})} \frac{1}{\lambda^{3}} \mathbb{E}_{\mathbf{x}_{t} \sim p_{t}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left\| \hat{x}_{\theta}(\mathbf{x}_{t}; s(t), \sigma(t)) - \hat{x}_{\theta}(\mathbf{x}_{t_{i}}; s(t_{i}), \sigma(t_{i})) \right\|^{2} \mathrm{d}\lambda
\end{aligned}
$$
重要性采样中采样的表达式：
$$
\pi(\lambda) \sim \frac{1}{\lambda^{3}} \left[\frac{1}{c^{2} + \lambda^{2}} - \frac{1}{c^{2} + \lambda_{i}^{2}}\right]
$$
重要性采样中积分的表达式：
$$
\text{KLUB} \approx \frac{\left\| \hat{x}_{\theta}(\mathbf{x}_{t}; s(t), \sigma(t)) - \hat{x}_{\theta}(\mathbf{x}_{t_{i}}; s(t_{i}), \sigma(t_{i})) \right\|^{2}}{\frac{1}{c^{2} + \lambda^{2}} - \frac{1}{c^{2} + \lambda_{i}^{2}}}
$$

KLUB 计算代码：
```python
def estimate_klub(t_min, t_mid, t_max, model, dataloader):
    x0 = next(dataloader)

    nsr_min = sigma(t_min) / scale(t_min)
    nsr_mid = sigma(t_mid) / scale(t_mid)
    nsr_max = sigma(t_max) / scale(t_max)

    nsr = importance_sampling(nsr_min, nsr_mid, nsr_max)
    nsr_upper = nsr_mid if nsr_t < nsr_mid else nsr_max

    t = nsr_inv(nsr)
    t_upper = nsr_inv(t_upper)
    scale_t = scale(t)
    scale_t_upper = scale(t_upper)
    sigma_t = sigma(t)
    sigma_t_upper = sigma(t_upper)

    x_t = scale_t * x0 + sigma_t * torch.randn_like(x0)
    x_t_upper = (scale_t_upper * x_t + ((scale_t * sigma_t_upper) ** 2 - (scale_t_upper * sigma_t) ** 2) * torch.randn_like(x0)) / scale_t

    y_t = model(x_t, scale_t, sigma_t)
    y_t_upper = model(x_t_upper, scale_t_upper, sigma_t_upper)

    weight = 1 / (scale_t ** 2 + c ** 2) - 1 / (scale_t_upper ** 2 + c ** 2)
    klub = weight * (y_t - y_t_upper).square().sum([-1, -2])
```

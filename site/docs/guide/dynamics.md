# 动力学基础

::: tip 一句话总结
几乎所有“基于模型”的控制器都只用到刚体动力学方程 $M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau + \tau_{\text{ext}}$ 中的三项，以及雅可比 $J(q)$。本页给出它们的性质与高效算法，并说明本站如何在 GPU 上批量计算它们、如何在浏览器里从 MuJoCo 取得同样的量。
:::

## 关节空间动力学

$n$ 自由度串联机械臂的拉格朗日方程写成

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) + \tau_f(\dot q) = \tau + J(q)^\top F_{\text{ext}} ,
$$

- $M(q) \in \mathbb R^{n\times n}$：惯性矩阵（对称正定），本站包括电机转子折算到关节的惯量（armature）$I_a$，即 $M \leftarrow M + \operatorname{diag}(I_a)$；
- $C(q,\dot q)\dot q$：科氏力与离心力；
- $g(q)$：重力矩；
- $\tau_f$：关节摩擦（粘性 $D\dot q$ + 库仑 $f_c\,\operatorname{sign}(\dot q)$）；
- $F_{\text{ext}}$：作用在末端的外力旋量。

常把 $h(q,\dot q) = C(q,\dot q)\dot q + g(q)$ 称为**偏置力矩**。

### 控制设计中反复用到的四条性质

1. **$M(q)$ 对称正定且有界**：$0 < m_1 I \preceq M(q) \preceq m_2 I$，所以 $M^{-1}$ 总存在，计算力矩控制总是可行的。
2. **$\dot M - 2C$ 反对称**（当 $C$ 取 Christoffel 符号形式时）：$\dot q^\top(\dot M - 2C)\dot q = 0$。这是被动性：总能量的变化率等于输入功率，$\frac{d}{dt}\big(\tfrac12\dot q^\top M\dot q + U\big) = \dot q^\top\tau$。PD + 重力补偿的稳定性证明、Slotine–Li 自适应控制和动量观测器都依赖这一条。
3. **对惯性参数线性**：$M(q)\ddot q + C(q,\dot q)\dot q + g(q) = Y(q,\dot q,\ddot q)\,\pi$，其中 $\pi$ 由每个连杆的质量、一阶矩、惯性张量共 10 个参数组成。这是参数辨识和[自适应控制](/methods/robust-adaptive)的基础。
4. **$C(q,\dot q)\dot q$ 关于 $\dot q$ 是二次的**：$C(q,a)\,b$ 是 $(a,b)$ 的对称双线性形式（Christoffel 形式下）。

## 计算方法

### 递归牛顿–欧拉（RNEA）

逆动力学 $\tau = \mathrm{ID}(q,\dot q,\ddot q)$ 用 RNEA 在 $O(n)$ 时间内算出：

- 从基座到末端递推每个连杆的速度、加速度：$v_i = v_{i-1} + s_i\dot q_i$，$a_i = a_{i-1} + s_i\ddot q_i + v_i \times s_i\dot q_i$（空间向量记号，$s_i$ 为关节运动子空间，重力通过令基座加速度为 $-g$ 计入）；
- 再从末端到基座回推力：$f_i = I_i a_i + v_i \times^* I_i v_i + \sum_{c} f_c$，$\tau_i = s_i^\top f_i$。

于是 $g(q) = \mathrm{ID}(q, 0, 0)$，$h(q,\dot q) = \mathrm{ID}(q,\dot q,0)$；$M$ 的第 $j$ 列是 $\mathrm{ID}(q,0,e_j)$ 去掉重力（或用复合刚体算法 CRBA）。

### Christoffel 一致的 $C(q,\dot q)\,\dot q_r$ {#christoffel}

Slotine–Li 型控制器（PD+、滑模、自适应）需要的是 $M\ddot q_r + C(q,\dot q)\dot q_r + g$，其中 $\dot q_r \ne \dot q$。标准 RNEA 只能给出 $C(q,\dot q)\dot q$。Echeandia & Wensing（2021）给出一个修改版 RNEA：前向递推中用 $a_i = a_{i-1} + s_i\ddot q_{r,i} + v_i\times s_i\dot q_{r,i}$，后向递推中把 $v\times^* I v$ 换成

$$
B(I, v)\,v_r = \tfrac12\left(v\times^* I v_r - I(v\times v_r) + v_r\times^* I v\right),
$$

一次 $O(n)$ 递推即得 $M\ddot q_r + C(q,\dot q)\dot q_r + g$，并且对应的 $C$ 满足 $\dot M - 2C$ 反对称。本站的 <Src path="rczoo/dynamics.py" symbol="Dynamics.slotine_li" /> 就是这个算法，单元测试用自动微分算出的 Christoffel 符号逐项核对。

浏览器里没有这个修改版 RNEA，但可以利用性质 4（对称双线性）用标准 RNEA 做**极化**：

$$
C(q,\dot q)\,\dot q_r = \tfrac14\left[c(\dot q + \dot q_r) - c(\dot q - \dot q_r)\right],\qquad c(v) = h(q,v) - g(q).
$$

两次 `mj_rne` 调用即可（<Src path="site/docs/.vitepress/theme/lib/dynamics.ts" symbol="NominalModel.coriolis" />）。

### 负载回归矩阵

固连在末端连杆上的未知负载只改变最后一个连杆的 10 个惯性参数 $\pi = [m,\ m c,\ I_o]$（质量、一阶矩、对连杆原点的惯性张量）。它对关节力矩的贡献对 $\pi$ 线性：

$$
\Delta\tau = Y_\pi(q,\dot q,\dot q_r,\ddot q_r)\,\pi .
$$

<Src path="rczoo/dynamics.py" symbol="Dynamics.payload_regressor" /> 用 10 次单位参数的 Slotine–Li 递推组装出 $Y_\pi$，测试中与 MuJoCo 里真实加上负载的模型核对。

## 任务空间

TCP 位姿 $(p, R)$、几何雅可比 $J = [J_v; J_\omega]$（世界坐标系，线速度在前），$\dot x = J\dot q$，$\ddot x = J\ddot q + \dot J\dot q$。姿态误差统一用

$$
e_R = \operatorname{Log}(R_d R^\top) \in \mathbb R^3 ,
$$

即从当前姿态转到期望姿态的旋转向量（世界坐标系），位姿误差 $e = [p_d - p;\ e_R]$。

**任务空间惯量** $\Lambda(q) = (J M^{-1} J^\top)^{-1}$ 是末端“感受到”的惯性。冗余机械臂（Panda）的**动力学一致零空间投影** $N^\top = I - J^\top \Lambda J M^{-1}$ 保证零空间力矩不会在末端产生加速度（Khatib 1987）。接近奇异时 $J M^{-1}J^\top$ 病态，本站加 $10^{-3}I$ 正则化。

## 本站的实现

| 量 | Python（基准，GPU 批量） | 浏览器 |
| --- | --- | --- |
| $p, R, J$ | 解析正运动学 | `mj_kinematics` + `mj_jacSite` |
| $M(q)$ | RNEA 列（含 armature） | `mj_crb` + `mj_fullM` |
| $h, g$ | RNEA | `mj_rne`（$\dot q = 0$ 得 $g$） |
| $\dot J\dot q$ | 一次前向递推 | `mj_jacDot` |
| $C(q,\dot q)\dot q_r$ | Christoffel RNEA | 两次 `mj_rne` 极化 |

两边在 <Src path="site/test/dynamics.test.ts" /> 中逐项对比（$M$ 误差 $<2\times10^{-5}$，$g, h$ 误差 $<2\times10^{-4}$ N·m，差异来自 MJCF 导出时的数值舍入）。

浏览器端用的是**第二个** MuJoCo 模型实例（负载质量置零）只做模型查询，与被仿真的对象完全分开——控制器看到的永远是标称模型，就像在真实机器人上一样。

## 参考文献

- R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.
- S. Echeandia, P. M. Wensing, "Numerical methods to compute the Coriolis matrix and Christoffel symbols for rigid-body systems," *J. Comput. Nonlinear Dyn.*, 2021.
- J.-J. E. Slotine, W. Li, "On the adaptive control of robot manipulators," *IJRR*, 1987.
- O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation," *IEEE J. Robot. Autom.*, 1987.

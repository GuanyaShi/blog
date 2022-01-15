---
layout: post
title: Learning-theoretic Perspectives on MPC via Competitive Control
---

Since the 1980s, Model Predictive Control (MPC) has been one of the most influential and popular process control methods in industries. The key idea of MPC is straightforward: with a finite look-ahead window of the future, MPC optimizes a finite-time optimal control problem at each time step, but only implements/executes the current timeslot and then optimizes again at the next time step, repeatedly. Actually, the second part "_only implements the current timeslot and reoptimizes at each time step_" is one of the reasons MPC was not that popular before the 1980s --- iteratively solving complex optimal control problems at high frequency was such a luxury task before computational power took off. 

Here is a trajectory tracking problem to explain how MPC works (visualized in the figure below). Suppose a robot is tracking a trajectory $\{d_1,d_2,\cdots,d_T\}$. Assume the robot's dynamics follows $x_{t+1}=A_{t}x_{t} + B_tu_t + w_t, 1\leq t\leq T$, where $x_t,u_t,w_t$ are state, control input, and disturbance respectively. At time step $t$, the cost the robot needs to pay is $q_t\|x_t-d_t\|^2+\|u_t\|^2$ where $q_t$ is used to balance between the tracking cost and the fuel cost. At each time step the robot has $k$-step predictions of the future: $d_{t+1},\cdots,d_{t+k}$ (trajectory), $A_{t},\cdots,A_{t+k-1},B_{t},\cdots,B_{t+k-1}$ (dynamics), $w_{t},\cdots,w_{t+k-1}$ (disturbance), and $q_{t+1},\cdots,q_{t+k}$ (cost functions). In this blog we use $I_t$ to denote a set containing all these information at time step $t$.

![](https://i.imgur.com/rbLkB1O.png)
 
It is how MPC works: at each time step, it solves the following $k$-step optimal control problem:
$$
\begin{align}
& \min_{u_t,\cdots,u_{t+k-1}} \sum_{i=1}^{k}\left( q_{t+i}\|x_{t+i}-d_{t+i}\|^2 + \|u_{t+i-1}\|^2 \right) + Q(x_{t+k}) \\
\mathrm{s.t.} \quad & x_{t+i+1} = A_{t+i}x_{t+i} + B_{t+i}u_{t+i} + w_{t+i}, \quad 0\leq i\leq k-1
\end{align}
$$

where $Q(\cdot)$ regularizes the terminal state. Having a proper $Q$ is critical for the stability and performance of MPC. Suppose the optimal solution from the above optimization problem is $\bar{u}_{t,t},...,\bar{u}_{t+k-1|t}$ and $u^*_{t,t},...,u^*_{t+k-1,t}$. A key feature of MPC is that only the first solved action $u^{\mathrm{MPC}}_{t,t}$ is executed/used, and at the next step we need to solve another optimization problem for $u^{\mathrm{MPC}}_{t+1,t+1}$. 

**Remark 1**: This blog will focus on MPC problems _without constraints_. In practice there are two types of constraints used in MPC: state constraint (i.e., $x_t\in\mathcal{X}_t$, typically related to safety) and input constraint (i.e., $u_t\in\mathcal{U}_t$).

**Remark 2**: Note that the available information $I_t$ at each time step is given _in an online manner_, which could be adaptive or adversarial. For example, although $d_{t+1},\cdots,d_{t+k}$ is known at step $t$, the desired trajectory generator can immediately give an adversarial $d_{t+k+1}$ at the next step after observing MPC's action.

Although the idea of MPC seems simple, the theoretical analysis gets very involved very quickly. Interestingly, MPC has a similar history as deep learning in terms of the relationship between extraordinary practical performance and the theory behind it. In the beginning, MPC was more or less like magic --- it worked very well in practice but lacked theoretical explanations. Later on, thanks to a lot of great control theorists, MPC theory has been largely developed and many variants have been studied, such as nonlinear MPC (MPC with nonlinear dynamics), explicit MPC (pre-solve the optimization problem offline), and robust MPC (dealing with uncertain dynamics), to name a few.

However, the existing theory for MPC is mostly from control-theoretic perspectives, and the results focus on stability, robustness, and asymptotic convergence. MPC is still begging learning-theoretic understandings (e.g., finite-time regret bounds). In Ben Recht's blog post ["What We've Learned to Control"](http://www.argmin.net/2020/06/29/tour-revisited/) he said: 
> _So many theorists are spending a lot of time studying RL algorithms, but few in the ML community are analyzing MPC and why it’s so successful. We should rebalance our allocation of mental resources! ... I’d urge the MPC crowd to connect more with the learning theory crowd to see if a common ground can be found to better understand how MPC works and how we might push its performance even farther._

In this blog, we will discuss some recent results which took the first step in understanding MPC from learning-theoretic perspectives. More specifically, we will show that MPC is a _competitive online learner_ and it enjoys _near-optimal dynamic regret guarantees_. 

## MPC can be viewed as a greedy and receding-horizon online learner

As we pointed out in Remark 2, $I_t$ is revealed _in an online manner_. That means, MPC can be interpreted as an online learner: $\textrm{MPC}(I_{t})\rightarrow u_t$, which returns an action $u_t$ given $I_{1:t}$. Further, note that as an online learner, (standard) MPC has two important features:
* It is greedy/myopic. The program $\textrm{MPC}(I_{t})$ returns the optimal action in a short time window by solving a $k$-step optimal control problem.
* It is receding-horizon. Recall that although $I_t$ is sufficient to predict future $k$ steps, $\textrm{MPC}(I_{t})$ only implements the current timeslot solution.

## Beyond no-regret: competitive online control problems

With the online learning perspective comes an important question: is MPC a strong online learner with some guarantees?

To answer this question, one natural idea is to bound the suboptimality gap between MPC and a "clairvoyant" policy $\mathrm{OPT}(I_1,\cdots,I_T)$ which knows the full sequence $I_1,\cdots,I_T$ in advance. Since $\mathrm{OPT}$ knows all the information in hindsight, it can solve a $T$-step optimal control problem and achieves globally optimal performance. For example, $\mathrm{OPT}$ for the aforementioned trajectory tracking problem is given by:

$$
\begin{align}
& \min_{u_1,\cdots,u_{T}} \sum_{t=1}^{T} q_{t+1}\|x_{t+1}-d_{t+1}\|^2 + \|u_{t}\|^2 \\
\mathrm{s.t.} \quad & x_{t+1} = A_{t}x_{t} + B_{t}u_{t} + w_{t}, \quad 1\leq t\leq T
\end{align}
$$

Let's define $J(\mathrm{OPT},I_{1:T})$ as the total cost incurred by $\mathrm{OPT}$ with $I_{1:T}$, and similarly for MPC. Clearly $J(\mathrm{OPT},I_{1:T})\leq J(\mathrm{MPC},I_{1:T})$. To quantify the suboptimality gap, we consider two metrics:
* _Dynamic regret_: $\mathrm{DR}=\sup_{I_{1:T}}J(\mathrm{MPC},I_{1:T})-J(\mathrm{OPT},I_{1:T})$. In some literature dynamic regret is also called _competitive difference_.
* _Competitive ratio_: $\mathrm{CR}=\sup_{I_{1:T}}J(\mathrm{MPC},I_{1:T})/J(\mathrm{OPT},I_{1:T})$.

Note that the word "competitive" appears in both metrics (competitive difference and competitive ratio). We call online control problems pursuing guarantees using these two metrics _competitive online control problems_. Recently, considerable effort has been made in this region. For example, recently [Gautam Goel](https://gautamcgoel.github.io/) and Babak Hassibi characterized the structure of the [dynamic-regret-optimal](https://arxiv.org/pdf/2106.12097.pdf) and [competitive-ratio-optimal](https://arxiv.org/pdf/2107.13657.pdf) policies in LTV systems with quadratic costs, using operator-theoretic techniques from robust control. For more examples, we refer to an incomplete paper list in the reference section.

**Difference from no-regret control**. Another commonly used metric in online control problems is _(static) regret_. The goal of no-regret control is to minimize the (static) regret of the online controller to the best controller from some specific class, where the most popular choice is the linear policy class. The definition of regret typically looks like this: $J(\mathrm{MPC}) - \min_{\pi\in\Pi}J(\pi)$, where $\Pi$ is the policy class known as a priori.

However, in many control problems, it is _extremely hard_ to select a reasonable comparator class $\Pi$ for regret analysis. For example, the MPC problem in this blog, nonlinear systems, and time-variant systems. It is because, for those systems, the structure of the optimal policy is much more complicated and intractable than LQR problems (where we know the optimal controller is linear). Therefore, instead of considering a specific comparator class, competitive online control directly offers _global optimality_ guarantees. In fact, choosing the linear controller comparator class could give us an arbitrarily large suboptimality gap (even in very simple systems, see an example [here](https://arxiv.org/abs/2002.05318)) --- it means that no-regret controllers against the best linear policy could still perform poorly.

## MPC is a competitive online learner

Finally, we are ready to answer the question posed at the beginning: is MPC a strong online learner?

**Standard LQR problems with adversarial disturbance**. In this case, the dynamics is $x_{t+1}=Ax_t+Bu_t+w_t$ ($w_t$ could be adversarial) and the cost function is $x_t^\top Qx_t+u_t^\top Ru_t$. In paper [[1](https://arxiv.org/pdf/2006.07569.pdf)], we proved two key results:
* For any LQR system, with $k$-step predictions of $w_t$, MPC's dynamic regret is $O(\lambda^kT+1)$ where $0\leq\lambda<1$ is some constant depends on the system parameters.
* There exist some LQR systems such that the dynamic-regret-optimal policy's dynamic regret is $\Omega(\lambda^k(T-k))$.

These results show the power of prediction in online control --- MPC only needs $O(\log T)$ predictions to achieve a constant dynamic regret, and the lower bound is also $\log T$! In other words, MPC is near-optimal in terms of dynamic regret. The numerical results match the theory very well. As shown in the figure below, in a trajectory tracking problem, as the number of predictions increases, the performance of MPC (in terms of its dynamic regret) improves _exponentially_.

![](https://i.imgur.com/G4gsC4M.png)

**LTV systems with general strongly convex cost functions**. In this case, the dynamics follows $x_{t+1}=A_tx_t+B_tu_t+w_t$ and the cost function is in a general form $f_t(x_t)+c_t(u_t)$. In paper [[2](https://proceedings.neurips.cc/paper/2021/file/298f587406c914fad5373bb689300433-Paper.pdf)], we also proved two key results:
* MPC's dynamic regret is $O(\lambda^kT+1)$
* MPC's competitive ratio is $1+O(\lambda^k)$

Again, these results imply that MPC only needs $\log T$ predictions to be 1-competitive or achieve a constant dynamic regret.

**Inexact predictions**. Note that the above results focus on the _exact prediction_ case, where $I_t$ is perfectly revealed at step $t$. What if only an inexact version $\hat{I}_t$ is revealed? Fortunately, the standard MPC approach is still competitive, but a residual term will be added to the competitive ratio/difference which depends on how close $\hat{I}_t$ is to $I_t$ (see more discussion in paper [[3](https://arxiv.org/abs/2010.11637)] and [this paper](https://arxiv.org/pdf/2102.01309.pdf) from Na Li's group). Moreover, intuitively, the standard MPC may not be the best approach in this case --- suppose the prediction $\hat{I}_t$ is awful, we have no reason to fully trust it. Based on this intuition, we proposed an adaptive and robust predictive controller in paper [[4](https://arxiv.org/pdf/2106.09659.pdf)].

## Takeaway: why is MPC competitive?
From an online theory's perspective, it is not that intuitive why the specific MPC strategy is competitive, especially considering that MPC is greedy and myopic. For example, in online learning, some greedy policy such as Follow the Leader (FTL) fails in the worst case, and the regularized version (Regularzied FTL, RFTL) enjoys strong regret guarantees even in the worst case (see Elad Hazan's [OCO book](https://arxiv.org/pdf/1909.05207.pdf) for an introduction). Given this, there must be some fundamental properties from the dynamical system and some algorithmic principles from MPC which allow a greedy policy competitive. Understanding those properties and principles will significantly help us design new decision-making algorithms. We haven't fully understood them, but here are some key points:
* **The power of predictions.** The structure of the online optimal control problem makes the power of prediction "_exponential_". In other words, the adversarial player has exponentially decaying power to hurt the system as the number of predictions increases. 
* **Receding horizon is crucial.** The receding horizon strategy seems quite expensive, but it is crucial for the guarantees we presented. 
* **Closed-loop stability to algorithmic stability.** The proofs in papers [[1](https://arxiv.org/pdf/2006.07569.pdf)] and [[2](https://proceedings.neurips.cc/paper/2021/file/298f587406c914fad5373bb689300433-Paper.pdf)] highly rely on the closed-loop stability of the dynamical systems. In particular, we "translated" closed-loop stability to algorithmic stability which ensures that both the policy output and the system response will not change drastically if the input is perturbed a little bit. Interestingly, this intuition matches the design philosophy of RFTL, even though the algorithmic stability of RFTL is from a clever design of the algorithm itself while for MPC it is from receding horizon strategy and the structure of the underlying dynamical systems.

## Acknowledgements
Thanks to [Yiheng Lin](https://yihenglin97.github.io/), [Yisong Yue](http://www.yisongyue.com/) and [Adam Wierman](https://adamwierman.com/) for feedback on this post.

## References

* Some recent competitive online control papers (a very incomplete list) [[a](https://arxiv.org/pdf/2002.05318.pdf),[b](https://arxiv.org/pdf/2106.12097.pdf),[c](https://arxiv.org/pdf/2107.13657.pdf),[d](https://arxiv.org/pdf/2111.00095.pdf),[e](https://arxiv.org/pdf/1906.11378.pdf),[f]()]
* [[1](https://arxiv.org/pdf/2006.07569.pdf)] The Power of Predictions in Online Control (NeurIPS'20) 
* [[2](https://proceedings.neurips.cc/paper/2021/file/298f587406c914fad5373bb689300433-Paper.pdf)] Perturbation-based Regret Analysis of Predictive Control in Linear Time Varying Systems (NeurIPS'21)
* [[3](https://arxiv.org/abs/2010.11637)] Competitive Control with Delayed Imperfect Information (preprint)
* [[4](https://arxiv.org/pdf/2106.09659.pdf)] Robustness and Consistency in Linear Quadratic Control with Predictions (SIGMETRICS'22)

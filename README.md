# Calculating $\pi$ (pi) 
[Buffon's needle problem](https://en.wikipedia.org/wiki/Buffon%27s_needle_problem) is one of the earliest problems in probabilistic geometry. The experiment under specific conditions allows us to estimate the value of $\pi$ (pi). 
For this, set up a striped background of stripe length $l$ and some needles $n$ of the same length. Scatter the needles as _randomly_ as possible on the striped background (see image below) and count the number of needles (_h_) crossing the stripe boundaries.

![](example_image.jpg)

Then,

$$
\pi \sim \frac{2n}{h}.
$$

### Python Scripts

Run both the scripts, i.e. ```buffon_needle_experiment_running.py``` and ```buffon_plot_running.py``` from **different terminals**.

For example:

```console
python buffon_needle_experiment_running.py
```

- Running ```buffon_needle_experiment_running.py``` requests for an input - image of the experiment. Then, analyses it, counts the number of needles crossing the stripe boundaries, and returns an estimate for $\pi$.
    - The program uses functions from ```scikit-image``` package for image analysis.
    - **NOTE:** Parameters may need adjustment. 
- Running ```buffon_plot_running.py``` opens a plot, automatically gets the estimated $\pi$ value, and displays it.
- The two programs communicate using MQTT protocol (package ```paho-mqtt```).
- Repeat the experiment as many times as possible to improve your estimate of $\pi$.



# Maths-CS-&-ML-Papers
A collection of my favourite papers (in no particular order) I have read concerning mathematics, computer science and machine learning. The analysis of each paper follows a structure resembling: 
- Title and authors
- Context/history
- Problems adressed
- Main contributions
- Strengths
- Weaknesses or limitations
- My takeaways.

# Wide-Impact Papers
### A collection of papers which had paradigm-shifting effects in research and or industry, with legendary status.
[COMPUTING, MACHINERY AND INTELLIGENCE](https://doi.org/10.1093/mind/LIX.236.433) (Alan Mathson Turing - 1950) A paper lightyears ahead of its time by the one and only **Alan Turing** - the father of modern computer science. Completed 14 years after his paper outlining Turing Machines and just a few years after his groundbreaking work on breaking Enigma at Bletchley Park.

Considering the field didn't formally exist yet, there were of course no real outstanding theoretical problems in computer science, hence the paper takes a more philosphical approach whilst also explaining the practicality of machine intelligence. Turing considers the question 'Can machines think?' taking a clear and logical tone to expertly defining the intricacies of _**"The Imitation Game"**_ - the test in which computers must successfully convince an interrogator that it is human. He then proceeds to define 9 objections to his 5 founding sections which define The Imitation Game and entail a brief history of the Analytical Engine and a general purpose computer.

Section 7 _Learning Machines_ is arguably the birth of machine learning and describes many modern machine and deep learning paradigms and models. Turing defines a 'child-programme' which must be taught by an 'education process', which essentially is the premise for modern day deep learning - the 'child-programme' being a model architecture and optimising a set of weights through gradient descent of a loss function - the 'education process'! He then clearly speaks of reinforcement learning when terms of 'rewards and punishments' are said to be applied to this child.

Turing states "An important feature of a learning machine is that its teacher will often be largely ignorant of quite what is going on inside, although he may still be able to some extent predict his pupil's behaviour" describing the black-box nature of deep learning and the fact that it is very hard (or impossible!) to derive by hand, the results of a deep neural network (DNN). However, as big a problem this 'untrustworthy' nature is today in fields which would benefit from machine learning (i.e. medicine), Turing instead completely trusts the machine at this point. It is important to note that our DNNs do not fit Turings descriptions as he mentions "Intelligent behaviour presumably consists in a departure from the completely disciplined behaviour involved in computation, but a rather slight one" implying that he envisioned a [system 2](https://thedecisionlab.com/reference-guide/philosophy/system-1-and-system-2-thinking) thinking model which has the ability to reason.

_**"We can only see a short distance ahead, but we can see plenty there that needs to be done."**_ is a powerful closing statement to the paper which set the world off on an adventure to reach a truly intelligent machine. Turing's vision for the field is truly astounding. While a lot of progress has been made in the field, we are yet to reach the materialisation of some of his most brilliant ideas within this paper, i.e the system 2 thinking Machine, written almost 75 years ago.

---


[ImageNet classification with deep convolutional neural networks](https://doi.org/10.1145/3065386) (Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton - 2012) The paper is synonymous with the modern deep learning revolution, it demonstrated the power of deep convolutional neural networks (CNNs), achieving performance on the ImageNet dataset previously unheard of. AlexNet was key in sparking widespread interest and research funding investments in deep learning.




[Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762) (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin) The paper responisble for the LLMs today, the transformer architecture has been a tremendous leap in natural language processing (NLP).


[Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) (Sepp Hochreiter, Jürgen Schmidhuber)
A highly influential architecture which eventually led to the transformer (from paper above), a variant of a recurrent neural network (RNN), but with a memory gate

%[]() (John von Neumann)


[A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (Claude Shannon)
In this paper Shannon establishes the field of Information Therory (there were earlier works in the 1920's). I have personally first encountered Shannon's work in an [ecology data analysis project](https://github.com/JIC1444/Spatial-Analysis-of-Microorganisms-Scandes) where his equation for _Shannon's Diversity index_ is derived from his equation in entropy, defined in this very paper.

The effects of this paper were felt in many fields, of course Information Theory may not exist in the same way without Shannon, but also the equation (below) defined in the paper reaches many corners of science: machine learning (loss functions and descision trees), ecology (diversity index), data compression algorithms, telecommunications (minimum number of bits without loss), cryptography (unpredicability of keys), thermodynamics and statistical physics (physical entropy), bioinformatics (analyses complexity of DNA and proteins, information flow in biological systems), NLP (uncertainty in text data for LLMs), finance (uncertainty in markets and markov chains), quantum computing (uncertainty and information flow in quatnum systems), neuroscience (information processing and entropy in neural systems) and signal processing (measures noise and optimises flow)

<img src="https://github.com/user-attachments/assets/6c3b2d79-dd83-494d-afc2-10f2b8820d6c" width="20%" height="20%">


The paper is its mathematical density and abstract nature, making it a difficult task to digest and understand the brilliance held within.

```{math}
$\sum p_i log(p_i)$
```


# Meta-Learning Papers
[Meta-Learning in Neural Networks: A Survey](https://arxiv.org/pdf/2004.05439) (Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey)

Hospedales et. al describe Meta-learning as a potential candidate to combat the data inefficiencies, poor knowledge transfer and unsupervised learning aspects of DNNs in research at the moment. There are different interpretations of the phrase 'meta-learning' but the paper focuses on **contemporary *neural network* meta-learning**. Meaning "algorithm learning, but focus on where this is achieved by end-to-end learning of an explicitly defined objective function (such as cross-entropy loss)".


# Dissertation Papers (ATT-GCN-LSTM / COVID19)
### This section reads in a rough order of increasing complexity (currently unfinished)


[Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (Guo S, Lin Y, Feng N, Song C, Wan H) 

[Short-Term Multi-Horizon Line Loss Rate Forecasting of a Distribution Network Using Attention-GCN-LSTM](https://arxiv.org/abs/2312.11898) (Liu J, Cao Y, Li Y, Guo Y, Deng W) 

[Integrating LSTMs and GNNs for COVID-19 Forecasting](https://arxiv.org/abs/2108.10052) (Sesti N, Garau-Luis JJ, Crawley E, Cameron B) 

[Predicting COVID-19 positivity and hospitalization with multi-scale graph neural networks](https://doi.org/10.1038/s41598-023-31222-6) (Skianis K, Nikolentzos G, Gallix B, Thiebaut R, Exarchakis G)


[Attention-based LSTM predictive model for the attitude and position of shield machine in tunneling](https://www.sciencedirect.com/science/article/pii/S2467967423000880) (Kang Q, Chen EJ, Li ZC, Luo HB, Liu Y)

















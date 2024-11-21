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

---

[COMPUTING, MACHINERY AND INTELLIGENCE](https://doi.org/10.1093/mind/LIX.236.433) (**Alan Mathison Turing - 1950**) A paper lightyears ahead of its time by the one and only **Alan Turing** - the father of modern computer science. Completed 14 years after his paper outlining Turing Machines and just a few years after his groundbreaking work on breaking Enigma at Bletchley Park.

Considering the field didn't formally exist yet, there were of course no real outstanding theoretical problems in computer science, hence the paper takes a more philosphical approach whilst also explaining the practicality of machine intelligence. Turing considers the question 'Can machines think?' taking a clear and logical tone to expertly defining the intricacies of _**"The Imitation Game"**_ - the test in which computers must successfully convince an interrogator that it is human. He then proceeds to define 9 objections to his 5 founding sections which define The Imitation Game and entail a brief history of the Analytical Engine and a general purpose computer.

Section 7 _Learning Machines_ is arguably the birth of machine learning and describes many modern machine and deep learning paradigms and models. Turing defines a 'child-programme' which must be taught by an 'education process', which essentially is the premise for modern day deep learning - the 'child-programme' being a model architecture and optimising a set of weights through gradient descent of a loss function - the 'education process'! He then clearly speaks of reinforcement learning when terms of 'rewards and punishments' are said to be applied to this child.

Turing states "An important feature of a learning machine is that its teacher will often be largely ignorant of quite what is going on inside, although he may still be able to some extent predict his pupil's behaviour" describing the black-box nature of deep learning and the fact that it is very hard (or impossible!) to derive by hand, the results of a deep neural network (DNN). However, as big a problem this 'untrustworthy' nature is today in fields which would benefit from machine learning (i.e. medicine), Turing instead completely trusts the machine at this point. It is important to note that our DNNs do not fit Turings descriptions as he mentions "Intelligent behaviour presumably consists in a departure from the completely disciplined behaviour involved in computation, but a rather slight one" implying that he envisioned a [system 2](https://thedecisionlab.com/reference-guide/philosophy/system-1-and-system-2-thinking) thinking model which has the ability to reason.

_**"We can only see a short distance ahead, but we can see plenty there that needs to be done."**_ is a powerful closing statement to the paper which set the world off on an adventure to reach a truly intelligent machine. Turing's vision for the field is truly astounding. While a lot of progress has been made in the field, we are yet to reach the materialisation of some of his most brilliant ideas within this paper, written almost 75 years ago.

---

[ImageNet classification with deep convolutional neural networks](https://doi.org/10.1145/3065386) (**Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton - 2012**) The paper is synonymous with the modern deep learning revolution, it demonstrated the power of deep convolutional neural networks (CNNs), achieving performance on the ImageNet dataset previously unheard of. AlexNet was key in sparking widespread interest and research funding investments in deep learning and the authors of the paper are still making huge strides in deep learning, notably Geoffrey Hinton won the 2024 Nobel Prize in Physics and Ilya Sutskever was chief scientist at OpenAI.

The model is of course by today's standards quite shallow, with 5 layers however the reason for the groundbreaking performance is highlighted in the prologue - the computational power and the significant amount of labelled data simply wasn't availiable in the 1980's, when it was deemded that training neural networks initialised with random wieghts was too monumentous of a task. Many famous techniques for deep learning were outlined in this paper - namely **dropout**, **ReLU** and **scale** (while not strictly a technique it is arguably the most important drivers of the most powerful models today). Additionally, the paper follows a logical flow and is nice to read visually, a great introductory paper for those just getting into the field of deep learning.

---

[Attention is all you need](https://doi.org/10.48550/arXiv.1706.03762) (**Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin - 2017**) The Google DeepMind paper responisble for the architecture dominant in deep learning today, the transformer architecture has been a tremendous leap in natural language processing (NLP) and many other fields.

"Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence", essentially looking at the elements which heavily influence the results in the data. The architecture's strengths lie partly in the encoding (and decoding) of the data into 512-dimensional vectors - this allows similar elements of the data to lie close to one another, allowing complex pattern recognition to occur, similar to a principal components analysis (PCA). However, these vectors are dynamic - changing as the model learns, differentiating it from the PCA.

Another key component of the architecture is the _**scaled dot-product attention**_, consisting of _queries, keys and values_. A _query_ can be thought of one token 'questioning' the other tokens relationship to it. A _key_ can be thought of a descriptor that a query can be answered with and finally a _value_ is the information returned after the attention process, giving context to the token in relation to others.

<img width="396" alt="Screenshot 2024-11-21 at 20 47 54" src="https://github.com/user-attachments/assets/3f06679b-5aae-46a9-8b5b-4a7a1e55fb4c">

where Q, K and V are a matricies of queries, keys and values respectively and sqrt(d_k) is the scaling factor, where d_k is the dimension of the keys vector.

The paper then goes on to justify its use of self-attention over recurrent or convolutional layers - overall the time complexity per layer is lower (n * n * d vs n * d * d, where d is equal to or greater than n). Also its ability to have the operations be sequentially excecuted. And finally the path length between long-range dependencies is infinitely shorter O(n) vs _just O(1)!_. The table comparing the different layers is displayed below with the caption.

<img width="916" alt="Screenshot 2024-11-21 at 21 11 34" src="https://github.com/user-attachments/assets/eff34ad5-b7b8-40f7-92a9-2b2857b2e38f">

Overall, a great paper, however the query, keys and values concepts are slightly harder to digest at first and takes a few reads to fully grasp. 

---

[Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) (**Sepp Hochreiter, Jürgen Schmidhuber - 1997**)
A highly influential architecture which eventually led to the transformer (from paper above), a variant of a recurrent neural network (RNN), but with a memory block, which significantly enhanced performance.

---

[A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (**Claude Shannon - 1948**)
In this paper Shannon establishes the field of Information Therory (there were earlier works in the 1920's). I have personally first encountered Shannon's work in an [ecology data analysis project](https://github.com/JIC1444/Spatial-Analysis-of-Microorganisms-Scandes) where his equation for _Shannon's Diversity index_ is derived from his equation in entropy, defined in this very paper.

The effects of this paper were felt in many fields, of course Information Theory may not exist in the same way without Shannon, but also the equation (below) defined in the paper reaches many corners of science, to name a few: 
  - machine learning (loss functions and descision trees),
  - ecology (diversity index),
  - data compression algorithms,
  - telecommunications (minimum number of bits without loss),
  - cryptography (unpredicability of keys),
  - thermodynamics and statistical physics (physical entropy),
  - bioinformatics (analyses complexity of DNA and proteins, information flow in biological systems),
  - NLP (uncertainty in text data for LLMs),
  - finance (uncertainty in markets and markov chains),
  - quantum computing (uncertainty and information flow in quatnum systems),
  - neuroscience (information processing and entropy in neural systems) and 
  - signal processing (measures noise and optimises flow)

<img width="712" alt="Screenshot 2024-11-21 at 08 59 50" src="https://github.com/user-attachments/assets/65b99f39-3631-4b9b-b9da-78d9a61905d4">


*Why is this equation so prevalent in so many fields?* This is due to features of the equation as well as its general nature and its utility in measuring uncertainty. 
 1. It only requires a set of probabilities to find H.
 2. It can be applied to any probabalistic situation, prevalent in the real world.
 3. It weights states with respect to (w.r.t) their probability, therefore can be used to see which states hold the most impact.
 4. It is widely adaptable to other fields - due to its simplicity, non-mathematicians can use this.

The paper runs deep with dense, abstract mathematics, making it a difficult task to digest and understand the brilliance held within, not to mention the paper is 55 pages in length. Any reader not directly studying Information Theory is reccomended to skip the mathematical proofs and read the introductory paragraphs to each section as well as the discussion.

---

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (**Olaf Ronneberger, Philipp Fischer, Thomas Brox - 2015**) A paper well-renouned 

It adresses the bigger and bigger dataset and parameters appearing in models after the release of the 2012 AlexNet paper, the U-Net focuses on data quality, more specifically, creating a segmentation mask using different class values for each group/object a pixel in an image belongs to. For example, imagine the view out of a car's windshield, you see a car, a bus and the road, all pixels in this mental screenshot are either of class value 0 (car), 1 (bus) or 2 (road) and perhaps 3 (the sky).

The paper finds that using such segmentation masks, as well as the genius U-shaped architecture. The image is first downsampled (encoded), reducing the spatial dimensions, increasing the feature dimensions. Then the upsampling (decoding) begins after the bottleneck at the trough of the U-Net. The skip connections allow the higher resolution

<img width="728" alt="Screenshot 2024-11-21 at 21 35 46" src="https://github.com/user-attachments/assets/0f7fc54f-f946-47a4-8c8c-98282b8ad47c">

The authors of the paper found very good results on a small dataset, with elastic deformation augmentations of images. The wide applications within biomedical image segmentation as well as extending to applications in self-driving cars and regular image segmentations.

---

# Meta-Learning Papers

---
[Meta-Learning in Neural Networks: A Survey](https://arxiv.org/pdf/2004.05439) (Timothy Hospedales, Antreas Antoniou, Paul Micaelli, Amos Storkey)

Hospedales et. al describe Meta-learning as a potential candidate to combat the data inefficiencies, poor knowledge transfer and unsupervised learning aspects of DNNs in research at the moment. There are different interpretations of the phrase 'meta-learning' but the paper focuses on **contemporary *neural network* meta-learning**. Meaning "algorithm learning, but focus on where this is achieved by end-to-end learning of an explicitly defined objective function (such as cross-entropy loss)".


# Dissertation Papers (ATT-GCN-LSTM / COVID19)
### This section reads in a rough order of increasing complexity (currently unfinished)


[Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (Guo S, Lin Y, Feng N, Song C, Wan H) 

[Short-Term Multi-Horizon Line Loss Rate Forecasting of a Distribution Network Using Attention-GCN-LSTM](https://arxiv.org/abs/2312.11898) (Liu J, Cao Y, Li Y, Guo Y, Deng W) 

[Integrating LSTMs and GNNs for COVID-19 Forecasting](https://arxiv.org/abs/2108.10052) (Sesti N, Garau-Luis JJ, Crawley E, Cameron B) 

[Predicting COVID-19 positivity and hospitalization with multi-scale graph neural networks](https://doi.org/10.1038/s41598-023-31222-6) (Skianis K, Nikolentzos G, Gallix B, Thiebaut R, Exarchakis G)


[Attention-based LSTM predictive model for the attitude and position of shield machine in tunneling](https://www.sciencedirect.com/science/article/pii/S2467967423000880) (Kang Q, Chen EJ, Li ZC, Luo HB, Liu Y)

















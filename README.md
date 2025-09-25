# ml-playground

This is a collection of miscellaneous data science and machine learning projects, documenting my own learning process.
Some projects are simple weekly assignments from university courses, others are personally motivated.

## contents

- [fashion-mnist-mlp](./fashion-mnist-mlp)
    - 2-layer perceptron, sigmoid hidden layer. ~86% test accuracy

- [nlp-zipf-gutenberg](./nlp-zipf-gutenberg)
    1. downloads top 20 most recently popular ebooks from Project Gutenberg
    2. preprocesses their textual content
    3. tokenizes and lemmatizes the text to build a vocabulary
    4. saves the top 100 most frequent words and their frequencies
    5. plots the frequencies with matplotlib
    6. also plots three different variations of the Zipf distribution, compared with the empirical data (linear and log-log scales)

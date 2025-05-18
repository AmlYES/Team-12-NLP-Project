# 2 QA Models with Retrieval

## The objective of that milestone was to experiment with 2 different models and compare the accuracies between them, utilizing the concept of retrieval question answering where the model goes to find the proper context to use it in  answering the question instead of directly giving it the context, both experiments were conducted on the squad_v2 QA dataset a subset of [5000] entries. 

(question) → [Retriever] → top-k relevant contexts → [Reader Model] → answer

Experiment 1: Roberta

Experiment 2: Fine Tuned T5

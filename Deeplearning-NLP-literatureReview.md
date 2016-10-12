# DL-NLP Meeting Week Apr 24 2016
## Overview
* NLP Topics, Applications and Trending 
* NLP Problems by different abstraction levels
* Suggested NLP topics to study deeper
* Selected Deep Learning for NLP topics
* Discussions
* Suggested Next Steps

## Detail
1. NLP Topics, Applications and Trending 
    * Word Tagging and Parsing
      - Part-of-Speech Tagging (POS): Mark word sequence with nouns, verbs, adjectives, adverbs, etc.
      - Semantic Role Labeling
      - Coreference Disambiguation: Finding all expressions that refer to the same entity in a text   
       For Example: Bill said he would come; the proper noun Bill and the pronoun he refer to the same person, namely to Bill
      - Parsing Tree
      - Word Sense Disambiguation (WSD): Identify the meaning by its context, e.g. I need new batteries for my MOUSE
    * Sentence/Document Level Tagging
      - Spam Filter/Spam Detection 
      - Sentiment Analysis : Product Reviews/Comments -> Good/Bad, or score of ratings
      - Machine Translation, Speech Recognition
      - Paraphraze: *Jason finished the job yesterday* vs. *The job was finished by Jason yesterday*
    * Information Extraction   
      - Named Entity Recognition (NER): Extract name, organization, location from a text
    * Intelligence Interaction: understanding languages in higher level
      - Question Answering (Hard)
      - Information Summarization (Hard): Summarize key information/Decision making process
      - Machine Dialog (Hard): Siri, etc


2. NLP Problems by different abstraction levels
    * _NLP is no magic, still falls into the domain of machine learning._
    * _The highest level/intelligence level, i.e. the most difficult, 
   would be a demonstration of real intelligence. The machine should understand text in high abstraction level, and know where to find needed information, and even how to response._  
      - Dialog, Summarizing Information, Question Answering 
    * _In Midium level/document level, most of them can be viewed as a tagging/recognition/classification problem. For example, semtiment analysis/spam filter just tried to tag each document as positive/negative; Speech recognition/Translation tried to tag the actual words for speech/another laguage. NER need to know if or not a word is a name, organization, etc._  
      - NER, Information Extraction, Translation, Speech Recognition, Sentiment Analysis, Spam Filter, Topic Modeling, etc
    * _In bottom level/Sentence and word level, NLP give each word/part of sentence a clear tag. The tag could be syntactic classes (nouns, verbs, etc), or parsing structure. Also, the tag could be true/false, in compariing two rephrazed sentences. Any good achievement in higher level would rely on efforts on lower level problems._ 
      - POS, Parsing, Coreference Disambiguation, Word Disambiguation, Semantic Role Labeling, Paraphrazing, etc
    * _In the most bottom level, we need reliable feature prepresentations: represent word and sentences into feature vectors_
      - Word representation problem. How to transform word into feature vectors? Very simple approach: Bag-of-Words, other feature based on deep learning is possible.     

3. Suggested NLP topics to study deeper
    * Sentiment Analysis is a good start. It is not highly intelligent, does not require hardware, has benchmark and widely used in business. 
    * NER, Information Extraction is also good. Very useful in business email conversations.
    * Parsing and POS is infrastracture. Good understanding on document level relies on high accuracy parsing structure. This is fundamental. 
    * Word representation using Deep Learning, also infrastracture
    * Machine Translation needs complete and sophisticated database. Speech Recognition needs hardware interface support and signal processing. Dialog and Question-Answering is out of our current capability. So these are not recommended.
    * [Optional] Topic Modeling
      - For topic modeling, there are not so many deep learning approaches.
      - David M. Blei is a top expert in this field; His model is not deep learning, still based on LDA - Latent Dirichlet Allocation but his result is state of the art.:  http://www.cs.columbia.edu/~blei/topicmodeling.html
      - Here are some references for deep learning topic model:
        - http://www.jmlr.org/proceedings/papers/v22/wan12/wan12.pdf
        - http://research.microsoft.com/en-us/um/beijing/events/kpdltm2014/Deep_Belief_Nets_for_Topic_Modeling.pdf
      - _Topic modeling is very useful in research paper searching, and news, or other large collections of documents. The performance evaluation for topic modeling is not clear, as it does not require as high accuracy as tagging or parsing. Topic model can be used for images, it is not within traditional NLP scope._
       
4. Selected Deep Learning for NLP topics
    * Word Representation, Word Embedding
      - Using RNN, MLP, CNN structure to learn features, as word representations
        - This is fundamental paper to help understand how NN can train a word representation and their performance in capturing semantic relationships: http://www.aclweb.org/anthology/N13-1090
        - This is more of a review on word representations, also mentioned NER: http://www.aclweb.org/anthology/P10-1040
        - This paper is a must read, it covers almost all NLP fundamental topics using their NN strcture. This paper will give a direct sense on how NN structure can be adjusted in NLP problems. http://arxiv.org/pdf/1103.0398v1.pdf
        - Thest two papers can be studied together. Although the latter one is not directly related to deep learning, it provides domain knowledge of word embedding problem in a unsupervised learning perspective. The first paper will provide insight into how supervised deep learning can train non-linear features.
          - http://ronan.collobert.com/pub/matos/2011_knowbases_aaai.pdf
          - http://www.cs.cmu.edu/~nasmith/papers/smith+eisner.acl05.pdf
        - This paper is important, though not related to NLP, in that it explains the necessity of unsupervised pre-training. It touches the core as the true advantage of deep structure is its ability to learn higher abstraction level features. So the true power of deep learning, though applied in many supervised tasks, lies in unserpervised side in the learning process. Suggest read this paper with RBM, Auto-Encoder chapters in Goodfella book. 
          - http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf
          - http://www.deeplearningbook.org/contents/autoencoders.html
       
        - [Optional] This is a popular tool for word representaton is word2vec. But it is not deep learning. Read this if we want a quick understanding of transforming word into feature vector based on computational lingustics approaches.
          - https://code.google.com/archive/p/word2vec/
       
    * POS, Parsing structure
       - State-of-the-art model is Recursive NN by Socher. Note that Recursive NN can also be applied to parse image, which could be important in image tagging and discription, but this is not our current focus because image requires many extra tricks to reach good performance.
          ![Recursive NN](https://www.aclweb.org/anthology/P/P14/P14-1105/image002.png)
       - These papers explaines Recursive-NN and applications in parsing. Focus on understanding how to DESIGN a deep learning structure for specific tasks. Like in here, NLP parsing is a tree strcture, and so their deep learning model also explores the recursive nature of a binary tree. 
         - http://jan.stanford.edu/pubs/2010SocherManningNg.pdf    
         - http://www.socher.org/uploads/Main/SocherBauerManningNg_ACL2013.pdf
         - Read with book chapter: http://www.deeplearningbook.org/contents/rnn.html
  
    * NER, Informarion Extraction
       - This survey is a comprehensive introduction to both traditional and deep learning approaches to NER. http://www.cfilt.iitb.ac.in/resources/surveys/rahul-ner-survey.pdf
       - This is the paper behind Stanford NER tools. It is not deep learning, but it adopts Conditional Random Field. http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf
       - There is no deep learning paper solely for the purpose of NER and Information Extraction. Because one deep NN can be generalized to multiple tasks, such as in http://arxiv.org/pdf/1103.0398v1.pdf. 
       
          
    * Sentiment Analysis
       - The state of the art is RNTN, Recursive Tensor Neural Network. Understanding this model requires understanding to Recursive-NN, and Socher's model for parsing. Also since it requires tensor back-propagation, the implementation is not easy. http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf
           
       - A fundamental tool for the old state of the art sentiment analysis, not deep learning, is Semi-Supervised learning based on EM-Multinomial Naive Bayes-Bag of Words. http://www.icml-2011.org/papers/93_icmlpaper.pdf   
       
    * Overall Language Modeling
       - This refers to probabilistic modeling. For example, how likely is the next word give the context, and what is the joint distribution of a sequence given the context? These models is useful in generative learning, like in machine translation and speech recognition, we need these distributions to select the most likely output. Also, this could be useful in generation text, like in dialog, and machine paper writing. 
       - The three papers are more of conceptual, if details are needed, should read its references. 
         - http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
         - http://www.fit.vutbr.cz/~imikolov/rnnlm/is2011_emp.pdf
         - http://www.aclweb.org/anthology/P12-1092                                 
                                        

5. Discussions                
    * I think to prepare for email dataset, auto-reply is not currently doable as it requires higher intelligence modeling like question-anwsering; [Comments] auto-reply is not possible and necessary, at least for now.
    * Something useful would be information extraction and NER. For example, extract information about price quoting, marketing, location, etc. 
    * Sentiment Analysis/Document Classification techniques are also useful, as we may want to classify email first into several categories before further processing.
    * It is hard to beat Stanford benchmark, but it is very helpful to understand their work on word embedding, parsing, NER and sentiment. Coupled with deep learning book, this will provide us a preliminary sense on how to design and make sense out of a network in future. 
    * This document is basically application oriented, a deeper and theoretical understanding of deep learning should come with continous reading and implementations.[小步快跑的发展模式]
    * [Comments] It is helpful to keep up with the latest trend, be aware of what the most brilliant people are doing, and what is the state-of-the-art in DL-NLP. 

6. Suggested Next Steps
    * Understand NLP Infrastructures: POS, Parsing, Labeling, using stanford course materials.
    * Understand Deep Learning Basics, especially R(recursive)NN and R(recurrent)NN and auto-encoders in book chapters. 
    * Read suggested papers
    * Test their and our implementations on sample data like movie review, etc.
    * __To be added ...__


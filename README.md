PROJECT LAMBDA
==============

A way of using two separate Differentiable Neural Computers (DNCs) to augment the performance of machine comprehension QA.

vanilla.py > access SquadObject that has been produced after running prosquad.py. this is where a vanilla DNC will be run

wordchar_embed.py > GloVe word embedding is concatenated with char-level embedding

prosquad.py > preprocessing of SQuAD documents

char_encode.py > Character-level embedding carried out here

char_embed.py + charmodel.py + train_char.py > Network through which character-level embedding is trained

models > Trained model is saved here

data > where SQuAD dataset is stored

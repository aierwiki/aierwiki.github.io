# Deep contextualized word representation

## What's the problem
- Learning high quality representations can be challenging. 
- They should ideally model both (1) complex characteristics of word use(e.g. syntax and sematics), and (2) how these uses vary across linguistic contexts(i.e., to model polysemy).

## How and Why
- Learned functions of the internal states of a deep bidirectional language model(biLM), which is pretrained on a large text corpus.
- Learn a linear combination of the vectors stacked above each input word for each end task.
- The higher-level LSTM states capture context-dependent aspects of word meaning while lower level states model aspects of syntax. Simultaneously exposing all of these signals is
highly beneficial, allowing the learned models select the types of semi-supervision that are most useful for each end task.
- Our approach also benefits from subword units through the use of character convolutions, and we seamlessly incorporate multi-sense information into downstream task without
explicitly training to predict predefined  sense class.
- **Bidirectional language models**: Given a sequence of N tokens, ($$t_1, t_2, ..., t_N$$), a forward language model computes the probability of the sequence by modeling the
probability of token $$t_k$$ given the history ($$t_1,...,t_{k-1}$$):
$$p(t_1, t_2, ..., t_N) = \prod_{k=1}^{N}p(t_k|t_1,t_2,...,t_{k-1})$$.
- Our formulation jointly maximizes the log likehood of the forward and backward directions:
$$\prod_{k=1}^N(log p(t_k|t_1,...,t_{k-1};\Theta^x,\overrightarrow{\Theta}_{LSTM},\Theta_s) 
+ (log p(t_k|t_{k+1},...,t_N;\Theta^x,\overleftarrow{\Theta}_{LSTM},\Theta_s)).$$
- We tie the parameters for both the token representation ($$\Theta_x$$) and Softmax layer ($$\Theta_s$$) in the forward and backward direction while maintaining separate
parameters for the LSTMs in each direction.
- **ELMo** is a task specific combination of the intermediate layer representations in the biLM.
- For each token $$t_k$$, a L-Layer biLM computes a set of $$2L + 1$$ representations
$$R_k = \{x_k^{LM}, \overrightarrow{h}_{k, j}^{LM},\overleftarrow{h}_{k, j}^{LM}|j=1,...,L\} = \{h_{k,j}^{LM}|j=0,...,L\}$$
- We compute a task specific weighting of all biLM layers:
$$ELMo_k^{task} = E(R_k;\Theta^{task})=\gamma^{task}\sum_{j=0}^Ls_j^{task}h_{k,j}^{LM}$$
- Considering that the activations of each biLM layer have a different distribution, in some cases it also helped to apply layer normalization to each biLM layer before weihting.
- To add ELMo to the supervised model, we first freeze the weights of the biLM and then concatenate the ELMo vector $$ELMo_k^{task}$$ with $$x_k$$ and pass the ELMo enhanced
representation $$[x_k;ELMo_k^{task}]$$ into the task RNN. As the remainder of the supervised model remains unchanged.
- Support joint training of both directions and add a residual connection between LSTM layers.


## Advantages
- Can be easily integrated in to existing models.
- significantly improves the state of the art of a broad range of NLP tasks.
- The biLM layers efficiently encode different types of synmatic and semantic information about words-in-context
- ELMo-enhanced models use smaller training sets more efficiently than models without ELMo.

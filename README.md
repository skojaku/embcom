# Community detection and Embedding

## Research questions:
- How well can graph embeddings capture communities in the SBM?
- What is the detectability threshold?
- Can we improve graph emebddings for community detection?

## Approach
- Focus on graph embeddings that implictly factorize some matrices
  - DeepWalk, node2vec, LINE [See Qiu et al. 2018](https://dl.acm.org/doi/10.1145/3159652.3159706) 
- Explore the spectral property of the matrices to derive the spectral properties 
  - [Nadakuditi and Newman 2012](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.188701), [Krzakara et al. 2013](https://www.pnas.org/content/110/52/20935)

## Results 
- Dense networks:
  - For fully connected graphs with multi-edges, the node2vec and DeepWalk can capture the community structure well [[note]](https://drive.google.com/file/d/1IR1FBy8NnYvhlytbxjpq5SZJxxOc7_zX/view?usp=sharing)
- Sparse networks:
  - Diminishing detectability limit? [[note]](https://drive.google.com/file/d/1o8LOYngQmNSWuivubWj8wBFpBzFLqIOl/view?usp=sharing)

## Todo

- Are the connected nodes closer than non-connected nodes in the embedding?
- Check the performance of the K-Means algorithm for large sparse networks

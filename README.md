# Kachow!
### "I am lightning" - Lightning McQueen

### How This Works
0) there are two models:
   - State constructor (get environment state from video)
   - control model (generates actions from environment state)
2) we train a custom [CNN (Convolutional Neural Network)](https://en.wikipedia.org/wiki/Convolutional_neural_network) + [LSTM (long short term memory)](https://en.wikipedia.org/wiki/Long_short-term_memory) hybrid model to reconstruct environment state from states encountered during control model training.
   - The environment state can be directly retreived from a simulated environment, but cant be collected from a physical one making this important
   - The LSTM allows us to remember past information, thus allowing us to remember where cubes are and where other robots are
   - The CNN allows us to get numeric data from image inputs, thus allowing us to input each frame captured by the camera to the LSTM
3) We train a control model based on SimBa[^1] using [PPO (Proximal Policy Optimization)](https://en.wikipedia.org/wiki/Proximal_policy_optimization)
   - This uses a custom environment and reward function

  TO DO:
   - make the environment
    - https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Create-New.md
   - verify that the code makes sense

 [^1]: [SimBa](https://arxiv.org/pdf/2410.09754)
 [^2]: [DIFFERENTIAL TRANSFORMER](https://arxiv.org/pdf/2410.05258)
 [^3]: [Decision Transformer](https://arxiv.org/pdf/2106.01345)
 [^4]: [Memory and Attention in Deep Learning](https://arxiv.org/pdf/2107.01390)
 [^5]: [Learning to Track with Object Permanence](https://openaccess.thecvf.com/content/ICCV2021/papers/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.pdf)
 [^6]: [Learning Object Permanence from Video](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610035.pdf)
 [^7]: [MambaOut RIP kobe](https://arxiv.org/pdf/2405.07992)
 [^8]: [ViViT](https://arxiv.org/pdf/2103.15691)
 [^9]: [SMART: SELF-SUPERVISED MULTI-TASK PRETRAIN-ING WITH CONTROL TRANSFORMERS](https://arxiv.org/pdf/2301.09816)

This a cross-lingual zero pronoun resolution using BERT and other positional features as well. 

Json files show we represent the data to our model. Json is formatted as:

  "id": example id

  "sentence1": sentence 1

  "sentence2": sentence 2 (if exists)

  "azp_index": [azp index in sentence 1, azp index in sentence 2], (-1 if does not exist)

  "candidate_start_indices": [[start indices of candidates in sentence 1],[start indices of candidates in sentence 2]]

  "candidate_end_indices": [[end indices of candidates in sentence 1],[end indices of candidates in sentence 2]]

  "labels": [candidate labels]

In train.json, we show how we represent one data instance when it appears as one sentence and when it appears as two sentences. 

OntoNotes 5.0 text is copyrighted, but freely available at https://catalog.ldc.upenn.edu/LDC2013T19


Note: We also included BERT tokenization file. We made it work smoothly with get_wordpiece_indices() function (check line 211 for more information).

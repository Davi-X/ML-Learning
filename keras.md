NOTE: `None` in any layer parameters means accept any number or form of that parameter
and Keras will automatically decide what value to use
## Text preprocessing
1. Tokenize (Split)
   1. Word level
   ```python
   # Get the most frequent n words
   tokenizer = Tokenizer(num_words=n)  
   ```
   2. Char level
   3. `N-grams`  
    One of the `feature engineering`s, Generally not using in NN, but quite powerful dor few layers light model e.g. logistic regression & Random forest
    ```md
    `The cat sat on the mat`
    ## 2-gram
    Set: {"The", "The cat", "cat", "cat sat", "sat on", "on", "on the", "the", "the mat", "mat}
    ```
```python
tokenizer.fit_on_texts(texts)
```
2. Encoding
   * One-hot
    ```python
    ## Without tokenizer
    # token_index : dict of each word with its index
    # max_len : max length of a sentence keep (word level)

    sentences = ['The cat sat on the mat', 'The dog ate my homework']

    results = np.zeros(shape=(len(sentences), max_length, max(token_index.values()) + 1))

    # Generally would think max_len restrict the value for the last dim of results, but actually not.
    # The last dim is the vocab_size
    
    ## Using Tokenizer
    # each sequence just keep the same size, not as one_hot matrix
    sequences = tokenizer.texts_to_sequences(sentences)
    one_hot_results = tokenizer.texts_to_matrix(sentences)
    ```

      * Hashing Trick
        Used when the number of unique tokens in your vocabulary is too large to handle explicitly.
        TODO
   * Word Embedding
    Instead of using 'binary, sparse, hardcoded' one-hot, using word embedding
    It is necessary to **lerarn** a new embedding space with every new task (Since many words' weights differentiate as tasks)
        * `Random embedding layer weight`, mainly depends on BP to adjust
        * Using `pre-computed model vocab` as Embedding layer weights (e.g. GloVe)
            useful when the training data is too small, cannot learn embedding properly
            Since the weights are pre-load, would not want BP to change its weights
            ```python
            model.layers[0].set_weights=([embedding_matrix])
            model.layers[0].trainable = False
            ```

    `Embedding` layer maps interger indices to dense vectors
     * Input : 2D tensor (batch_size, input_length)
     * Output: 3D tensor (batch_size, input_length, output_dim)

    Generally
    ```python
    # Since for Embedding layer, all sequences in a batch must have the same length
    # Introduce pad_sequences
    from keras.preprocessing.sequence import pad_sequences
    maxlen = n
    data = pad_sequence(data, maxlen=maxlen)
    Embedding(vocab_size, output_dim, input_length=maxlen)
    # `input_length` is needed if it followed with `Flatten` and `Dense`
    # Note that output_dim is a parameter need to tune
    ```
    Whatever `input_length` parameter is needed, cut sequences is needed

    Sometimes, following with a `Flatten` layer after `Embedding` layer to flatten to 2D tensor (batch_size, input_length * output_dim)

## Text Processing
1. Dense Layer only  
   Can only deal with single word or char, very limited  
   e.g. The movie is shit! It would treat it as a negative review (but possibly true in reality)
2. RNN + Dense
3. 1D CNN 

### RNN in Keras
__I assume you have understood RNN here__

For any RNN model:  
Each timestep actually is a Dense NN

`units`: number of hidden states of Dense NN for each timestep

#### Valina RNN : SimpleRNN  
* Input:   
  (batch_size, timesteps, input_features)
* Output: 
  * return_sequences = True  
  (batch_size, timesteps, output_features)

  * return_sequences = False(default)  
  (batch_size, output_features)

SimpleRNN does do perform well in handling with long sequences (e.g. texts)

#### LSTM & GRU
Due to the vanishing gradient problem, simpleRNN cannot remember some previous information

Understand details of RNN is not necessary, since they are just restrictions for choices but not engineering designs

LSTM performs better when dealing with harder NLP(e.g. Q&A, chatbot, Translation)

GRU(gated recurrent unit) works similar with LSTM but is a simplified version. Lower cost 

#### Dropout
One of the means to regularize data, as a parameter for layer
```python
model.add(layers.GRU(32,
        dropout=0.2,
        recurrent_dropout=0.2,
        input_shape=(None, float_data.shape[-1])))
# reucrrent_dropout is the drop out rate for single recurrent cell
```
Note that dropout network needs more time to converge, needs `larger number of epochs`

`Recurrent layer stacking` is useful 
TODO
# Leaderboards

👈 Choose a leaderboard on the left to see the results.


## 📊 How to Read the Leaderboards

The main score column is the `rank` column, showing the [mean rank
score](/methodology.md) of the model across all tasks. The lower the rank, the better the
model.

The columns that follow the `rank` column are metadata about the model:

- `num_model_parameters`: The total number of parameters in the model, in millions.
- `vocabulary_size`: The size of the model's vocabulary, in thousands.
- `max_sequence_length`: The maximum number of tokens that the model can process at a
  time.
- `commercially_licensed`: Whether the model can be used for commercial purposes. See
  [here](/faq.md) for more information.
- `merge`: Whether the model is a merge of other models.
- `speed`: The inference time of the model on a CPU, in tokens per second.

After these metadata columns, the individual scores for each dataset is shown. To read
more about the datasets, see the [datasets](/datasets) page. To read more about the
tasks, see the [tasks](/tasks) page. Lastly, if you're interested in the methodology
behind the benchmark, see the [methodology](/methodology) page.

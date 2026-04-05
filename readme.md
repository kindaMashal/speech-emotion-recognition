# Speech Emotion Recognition (SER) — CNN + LSTM

## Introduction

We are building a Speech Emotion Recognition model (SER) using CNN+LSTM. The data has been collected from 4 different audio sources: RAVDESS, CREMA-D, TESS, and SAVEE.

Our data is labeled as the following: 'neutral,' 'happy,' 'sad,' 'angry,' 'fear,' 'disgust,' or 'surprise.' Therefore, this is how our model should classify the emotions.

According to our findings by reading through multiple research papers and GitHub codes: a hybrid model of 1D CNNs + LSTMs are mostly used in audio related classification tasks. That doesn't mean, however, that it's the only method (as there are also 1D CNNs models only or LSTM+CNNs) but 1D CNN+LSTM showed promising results and high accuracy rates and thus we took that route.

## Environment

We decided to use Kaggle as our environment to train our model on — as it has a great virtual GPU (better than Google Collab's and Jupyter's virtual GPU) and it's free. Furthermore, we couldn't use our local device to train our model on (such as using Vscode) as none of us had a good GPU that is capable enough for our task.

In addition, considering we have large data to train our model with — Kaggle was the perfect place to store that data without it taking space from our online drive or local memory.

## 1D CNN vs 2D CNN

However we wanted to compare with 2D CNN as we weren't the ones who collected the data. In addition, most audio data is in 2D (Mel spectrograms) so why are some research papers opting for 1D?

## Results

Our experiment fell short. We didn't notice one important key in this Kaggle dataset the MFCC extraction function uses flatten=True by default, which means it has flattened the 2D MFCC matrix into 1D, so the model input is strictly 1D feature vectors. Therefore, we are extracting sequential features along the time axis and not over 2D spatial patterns.

```python
return np.squeeze(mfcc_result.T) if not flatten else np.ravel(mfcc_result.T)
```

This is why the 1D CNN+LSTM outperformed our 2D CNN+LSTM, the data was never truly 2D to begin with.

## Lessons Learned

It was a fun experiment to carry out regardless of our shortcoming to forget to visualize the dataset before building the model and seeing if that data is 2D or 1D. We live and we learn :)
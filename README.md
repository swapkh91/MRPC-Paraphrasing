# MRPC-Paraphrasing

Using MRPC dataset identify whether 2 sentences are similar in meaning or not. 

I've implemented ensemble model using xgboost and LSTM (Siamese LSTM). The LSTM implementation is inspired from [this](<https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07>) post. I have also created it as an API to directly query for similarity. The docker image can be found [here](https://hub.docker.com/r/swapkh91/firsttry91/tags). This is the first time I've worked with docker so please ignore naming conventions, it was just a try :)

The service expects a form JSON with key "injson". A sample input:
```
{"Sentence1": "PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .",
"Sentence2": "Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So ."}
```

Curl request:
```
curl --location --request POST 'http://localhost:4002/get_sentence_similarity' \
--form 'injson={"Sentence1": "PCCW '\''s chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .",
"Sentence2": "Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So ."}'
```

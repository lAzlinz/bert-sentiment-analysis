from transformers import pipeline

path_trained_model = './models/unbalanced_model/checkpoint-30630'
classifier = pipeline('sentiment-analysis', model=path_trained_model)

while True:
    headline = input('headline: ')
    if headline in ['q', 'quit', 'end']:
        break
    prediction = classifier(headline)[0]

    print(prediction)